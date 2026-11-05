"""The source replay probe's fixture builders and read-only oracles, node side.

``test_general_live_source_probe*.py`` build the reviewed fixtures the variants
use; here the same CLI (``probe.main``, over the shared ``FakeHost``) is held to
what a real node can hand it instead: a PCI function whose vendor/device cannot
be read, a root larger than the bound, an RDMA device without ports' state or a
driver link, refused mutation specs (unsafe names, a missing port or counter, a
tree over the device bound), a /proc/driver/nvidia without version or GPU list,
a /proc/meminfo with odd lines and missing fields -- and the read-only
``inventory``, ``audit-read`` and ``status`` over a tree with unsafe names, an
unparsable audit line and a window without fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_source_builders as builders
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional import test_general_live_source_probe as probe_tests

RUN = probe_tests.RUN


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> probe_tests.FakeHost:
    return probe_tests.make_host(tmp_path, monkeypatch)


def _main(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1])


def _refused(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    code, receipt = _main(capsys, *argv)
    assert code == 1 and receipt["error"] == "ProbeError", receipt
    return str(receipt["error_message"])


# --------------------------------------------------------------------------- #
# build-pci-fixture
# --------------------------------------------------------------------------- #
def _pci_root(tmp_path: Path, *functions: tuple[str, str | None, str]) -> Path:
    root = tmp_path / "sys" / "bus" / "pci" / "devices"
    root.mkdir(parents=True)
    for bdf, vendor, device in functions:
        slot = root / bdf
        slot.mkdir()
        if vendor is not None:
            (slot / "vendor").write_text(vendor + "\n")
        (slot / "device").write_text(device + "\n")
    return root


def test_a_pci_function_without_its_vendor_file_refuses_the_copy(
    host: probe_tests.FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _pci_root(
        tmp_path, ("0000:00:00.0", "0x10de", "0x2330"), ("0000:53:00.0", None, "0xefa2")
    )
    message = _refused(
        capsys, "build-pci-fixture", "--run-id", RUN, "--source", str(root)
    )
    assert message.startswith("0000:53:00.0: vendor/device unreadable:"), message
    assert not (probe.fixtures_dir(RUN) / "pci").exists(), "no half-built root"


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (("--efa-count", "33"), "efa-count is outside 0..32"),
        (("--source", "{tmp}/absent"), "absent is not a directory"),
        (("--source", "{root}"), "lists more than 3 functions"),
    ],
)
def test_pci_fixture_bounds_refuse_before_anything_is_left_behind(
    host: probe_tests.FakeHost,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
    needle: str,
) -> None:
    monkeypatch.setattr(builders, "MAX_PCI_ENTRIES", 3)
    root = _pci_root(
        tmp_path, *((f"0000:53:0{index}.0", "0x1d0f", "0xefa2") for index in range(4))
    )
    values = [item.format(tmp=tmp_path, root=root) for item in argv]
    message = _refused(capsys, "build-pci-fixture", "--run-id", RUN, *values)
    assert needle in message, message
    assert not (probe.fixtures_dir(RUN) / "pci").exists(), "no half-built root"


# --------------------------------------------------------------------------- #
# build-infiniband-fixture
# --------------------------------------------------------------------------- #
def _rdma_tree(tmp_path: Path, *, netdev: bool = True) -> tuple[Path, Path]:
    """rdmap0s0 complete (driver, net, counters); rdmap1s0 bare (no state, no driver)."""

    sysfs = tmp_path / "sys"
    function = sysfs / "devices" / "pci0000:50" / "0000:53:00.0"
    function.mkdir(parents=True)
    if netdev:
        (function / "net" / "eth1").mkdir(parents=True)
    (function / "uevent").write_text("DRIVER=efa\nPCI_SLOT_NAME=0000:53:00.0\n")
    (function / "vendor").write_text("0x1d0f\n")
    (function / "device").write_text("0xefa2\n")
    driver = sysfs / "bus" / "pci" / "drivers" / "efa"
    driver.mkdir(parents=True)
    (function / "driver").symlink_to(driver)
    source = sysfs / "class" / "infiniband"
    port = source / "rdmap0s0" / "ports" / "1"
    (port / "counters" / "nested").mkdir(parents=True)
    (port / "hw_counters").mkdir()
    (port / "state").write_text("4: ACTIVE\n")
    (port / "phys_state").write_text("5: LinkUp\n")
    (port / "counters" / "port_rcv_data").write_text("1024\n")
    (port / "hw_counters" / "rdma_read_bytes").write_text("7\n")
    (source / "rdmap0s0" / "device").symlink_to(function)
    bare = source / "rdmap1s0"
    (bare / "ports" / "1").mkdir(parents=True)
    (bare / "device").mkdir()
    (bare / "device" / "uevent").write_text("PCI_SLOT_NAME=0000:54:00.0\n")
    return source, sysfs / "class" / "net"


def test_the_copy_keeps_what_each_device_really_has_and_sets_a_new_driver(
    host: probe_tests.FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source, _ = _rdma_tree(tmp_path)
    code, built = _main(
        capsys,
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--driver",
        "rdmap1s0:ena",
    )
    assert code == 0, built
    assert built["devices"]["rdmap0s0"] == {
        "ports": {"1": {"state": "4: ACTIVE", "phys_state": "5: LinkUp"}},
        "counters": 2,
        "driver": "efa",
        "netdevs": ["eth1"],
    }, "a sub-directory under counters/ is no counter"
    assert built["devices"]["rdmap1s0"] == {
        "ports": {"1": {}},
        "counters": 0,
        "driver": None,
        "netdevs": [],
    }, "a port without state files and a device without a driver are copied bare"
    assert built["mutations"] == ["rdmap1s0/device/driver"]
    tree = Path(built["path"])
    uevent = (tree / "rdmap1s0" / "device" / "uevent").read_text()
    assert uevent == "PCI_SLOT_NAME=0000:54:00.0\nDRIVER=ena\n"
    assert (tree / "rdmap1s0" / "device" / "driver").resolve().name == "ena"


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (("--duplicate", "rdmap9s0"), "--duplicate must have 2 '='-separated parts"),
        (("--driver", ".hidden:none"), "unsafe RDMA device name '.hidden'"),
        (("--port-state", "rdmap0s0:x:4: ACTIVE"), "unsafe port name 'x'"),
        (("--port-state", "rdmap0s0:2:1: DOWN"), "device 'rdmap0s0' has no port '2'"),
        (("--driver", "rdmap0s0:EFA"), "unsafe driver name 'EFA'"),
        (
            ("--remove-counter", "rdmap0s0:1:stats/port_rcv_data"),
            "counter 'stats/port_rcv_data' is not allow-listed",
        ),
        (
            ("--remove-counter", "rdmap0s0:1:counters/link_downed"),
            "rdmap0s0:1:counters/link_downed is not in the tree",
        ),
    ],
)
def test_a_mutation_spec_outside_the_reviewed_shapes_is_refused(
    host: probe_tests.FakeHost,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: tuple[str, ...],
    needle: str,
) -> None:
    source, _ = _rdma_tree(tmp_path)
    message = _refused(
        capsys,
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        *argv,
    )
    assert needle in message, message
    assert not (probe.fixtures_dir(RUN) / "infiniband.json").exists(), (
        "a refused build writes no summary"
    )


def test_the_tree_source_must_exist_list_a_device_and_stay_bounded(
    host: probe_tests.FakeHost,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = ("build-infiniband-fixture", "--run-id", RUN, "--source")
    message = _refused(capsys, *build, str(tmp_path / "absent"))
    assert message == f"{tmp_path / 'absent'} is not a directory"
    empty = tmp_path / "empty"
    empty.mkdir()
    message = _refused(capsys, *build, str(empty))
    assert message == (
        f"{empty} lists no RDMA device; pass --allow-empty to build an empty tree"
    )
    source, _ = _rdma_tree(tmp_path)
    monkeypatch.setattr(builders, "MAX_TREE_DEVICES", 1)
    message = _refused(capsys, *build, str(source))
    assert message == "the fixture tree holds more than 1 devices"


# --------------------------------------------------------------------------- #
# build-nvidia-proc-fixture
# --------------------------------------------------------------------------- #
def _nvidia_proc(tmp_path: Path, *, version: bool = True, gpus: int = 2) -> Path:
    source = tmp_path / "proc" / "driver" / "nvidia"
    source.mkdir(parents=True)
    if version:
        (source / "version").write_text(
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05\n"
        )
    if gpus:
        for index in range(gpus):
            gpu = source / "gpus" / f"0000:{0x53 + index:02x}:00.0"
            gpu.mkdir(parents=True)
            (gpu / "information").write_text(f"Model: NVIDIA H200\nMinor: {index}\n")
    return source


def test_the_real_version_and_every_gpu_are_copied_when_nothing_is_cut(
    host: probe_tests.FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _nvidia_proc(tmp_path)
    code, built = _main(
        capsys, "build-nvidia-proc-fixture", "--run-id", RUN, "--source", str(source)
    )
    assert code == 0, built
    assert built["version"].endswith("580.95.05"), built
    assert built["gpus"] == ["0000:53:00.0", "0000:54:00.0"]
    copied = Path(built["path"]) / "gpus" / "0000:54:00.0" / "information"
    assert copied.read_text() == "Model: NVIDIA H200\nMinor: 1\n"


@pytest.mark.parametrize(
    ("layout", "argv", "needle"),
    [
        ({}, ("--version-text", "x" * 1025), "version text exceeds 1 KiB"),
        ({"version": False}, (), "/version is unreadable:"),
        ({"gpus": 0}, ("--omit-version",), "/gpus is unreadable:"),
        ({}, ("--gpu-count", "3"), "gpu-count is outside 0..<real GPU count>"),
    ],
)
def test_a_proc_tree_the_variant_cannot_be_cut_from_is_refused(
    host: probe_tests.FakeHost,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    layout: dict[str, Any],
    argv: tuple[str, ...],
    needle: str,
) -> None:
    source = _nvidia_proc(tmp_path, **layout)
    message = _refused(
        capsys,
        "build-nvidia-proc-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        *argv,
    )
    assert needle in message, message


# --------------------------------------------------------------------------- #
# build-meminfo-fixture
# --------------------------------------------------------------------------- #
def test_meminfo_keeps_odd_lines_and_appends_a_field_the_kernel_omits(
    host: probe_tests.FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "meminfo"
    source.write_text(
        "MemTotal:       2000000 kB\n"
        "MemFree:         500000 kB\n"
        "MemAvailable:   1000000 kB\n"
        "a line without a separator\n"
        "Unaccounted:\n"
        "DirectMap1G:    unknown kB\n"
    )
    code, built = _main(
        capsys,
        "build-meminfo-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--set",
        "SwapTotal=1000",
        "--set",
        "SwapFree=250",
    )
    assert code == 0, built
    lines = Path(built["path"]).read_text().splitlines()
    assert lines[3:6] == [
        "a line without a separator",
        "Unaccounted:",
        "DirectMap1G:    unknown kB",
    ], "lines the product would skip are kept verbatim"
    assert lines[6:] == [
        "SwapTotal:" + " " * 11 + "1000 kB",
        "SwapFree:" + " " * 12 + "250 kB",
    ], "a field the source lacks is appended in the kernel's 15-column format"
    assert "DirectMap1G" not in built["values_kb"], built["values_kb"]
    assert built["values_kb"]["SwapTotal"] == 1000.0
    assert (built["memory_used_percent"], built["swap_used_percent"]) == (50.0, 75.0)


def test_meminfo_refuses_an_unreadable_source_and_a_tree_in_its_place(
    host: probe_tests.FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build = ("build-meminfo-fixture", "--run-id", RUN, "--set", "MemFree=10")
    message = _refused(capsys, *build, "--source", str(tmp_path / "absent"))
    assert message.startswith(f"{tmp_path / 'absent'} is unreadable:"), message
    source = tmp_path / "meminfo"
    source.write_text("MemTotal: 100 kB\nMemFree: 50 kB\n")
    (probe.fixtures_dir(RUN) / probe.MEMINFO_FIXTURE).mkdir(parents=True)
    message = _refused(capsys, *build, "--source", str(source))
    assert message == "a fixture tree of that name exists"


# --------------------------------------------------------------------------- #
# The read-only oracles
# --------------------------------------------------------------------------- #
def test_the_inventory_skips_unsafe_names_and_a_device_without_its_function(
    host: probe_tests.FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # No netdev on the EFA function: the inventory then runs no ``ethtool -S``.
    source, netdevs = _rdma_tree(tmp_path, netdev=False)
    (source / ".rdmap-staging").mkdir()
    port = source / "rdmap0s0" / "ports"
    (port / "x").mkdir()
    (port / "1" / "counters" / "Not_A_Counter").write_text("1\n")
    loose = source / "rdmap2s0"
    (loose / "ports" / "1").mkdir(parents=True)
    netdevs.mkdir(parents=True)
    code, listed = _main(
        capsys,
        "inventory",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--netdev-root",
        str(netdevs),
    )
    assert code == 0, listed
    devices = listed["devices"]
    assert sorted(devices) == ["rdmap0s0", "rdmap1s0", "rdmap2s0"], devices
    assert sorted(devices["rdmap0s0"]["ports"]) == ["1"], "port 'x' is no port"
    assert devices["rdmap0s0"]["ports"]["1"]["counters"] == {"port_rcv_data": "1024"}
    assert devices["rdmap2s0"]["pci"] is None and devices["rdmap2s0"]["efa"] is False


def test_audit_read_refuses_other_tools_and_keeps_an_unparsable_line(
    host: probe_tests.FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    message = _refused(capsys, "audit-read", "--run-id", RUN, "--tool", "rm")
    assert message == "'rm' is not an allow-listed tool"
    code, empty = _main(capsys, "audit-read", "--run-id", RUN, "--tool", "smartctl")
    assert code == 0 and (empty["entries"], empty["calls"]) == ([], {}), empty
    audit = probe.audit_dir(RUN)
    audit.mkdir(parents=True)
    (audit / "smartctl.jsonl").write_text('{"argv": ["-a"]}\nnot json at all\n')
    code, read = _main(capsys, "audit-read", "--run-id", RUN, "--tool", "smartctl")
    assert code == 0, read
    assert read["entries"] == [{"argv": ["-a"]}, {"unparsable": "not json at all"}]


def test_status_needs_a_window_and_lists_trees_and_audit_lines(
    host: probe_tests.FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    message = _refused(capsys, "status", "--run-id", RUN)
    assert message == "no source replay window is open for this run"
    probe.save_state(
        RUN, {"run_id": RUN, "mode": "command", "wrappers": {"ipmitool": {}}}
    )
    code, bare = _main(capsys, "status", "--run-id", RUN)
    assert code == 0, bare
    assert (bare["fixtures"], bare["audit_lines"]) == ({}, {}), bare
    assert bare["wrappers"] == ["ipmitool"] and bare["unit_name"] == probe.HOST_UNIT
    code, built = _main(
        capsys, "build-pci-fixture", "--run-id", RUN, "--efa-count", "1"
    )
    assert code == 0, built
    audit = probe.audit_dir(RUN)
    audit.mkdir(parents=True)
    (audit / "ipmitool.jsonl").write_text('{"n": 1}\n{"n": 2}\n')
    code, listed = _main(capsys, "status", "--run-id", RUN)
    assert code == 0, listed
    assert listed["fixtures"]["pci"] == "<tree: 3 entries>", listed["fixtures"]
    assert listed["audit_lines"] == {"ipmitool": 2}
