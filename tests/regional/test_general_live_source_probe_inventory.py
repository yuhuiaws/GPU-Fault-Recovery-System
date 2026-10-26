"""Read-only oracle commands of the host-collector source replay probe:
``inventory`` and ``read-source`` (010 / 011 ``broken_pair``), split from
``test_general_live_source_probe`` (file size) on its ``FakeHost``. ``inventory``
lives in the builders file since 2026-10-26 and runs from the checkout's copy
here, as the ``_builder`` loader falls back to it."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional.test_general_live_source_probe import (
    RUN,
    FakeHost,
    _install,
    _main,
    _sysfs_infiniband,
    make_host,
)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHost:
    return make_host(tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
# read-only inventory and read-source (010 / 011 broken_pair)
# --------------------------------------------------------------------------- #
def _physical_netdevs(root: Path, *names: str) -> Path:
    net_root = root / "sys" / "class" / "net"
    for name in names:
        (net_root / name / "device").mkdir(parents=True)
    (net_root / "veth0").mkdir(parents=True)  # no device: not physical
    return net_root


def test_inventory_lists_the_tree_as_the_product_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _sysfs_infiniband(tmp_path / "host", devices=2)
    net_root = _physical_netdevs(tmp_path / "host", "eth0", "eth1", "ens5")
    listing = "NIC statistics:\n     queue_0_tx_cnt: 5\n     rnr_nak_err: 7\n"
    seen: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool = True, timeout: int = 120) -> Any:
        seen.append(list(command))
        return subprocess.CompletedProcess(command, 0, listing, "")

    monkeypatch.setattr(probe, "run", fake_run)
    monkeypatch.setattr(
        probe.shutil, "which", lambda tool, path=None: f"/usr/sbin/{tool}"
    )
    code, value = _main(
        "inventory",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--netdev-root",
        str(net_root),
        capsys=capsys,
    )
    assert code == 0, value
    device = value["devices"]["rdmap0s0"]
    assert device["efa"] is True and device["driver"] == "efa", device
    assert device["uevent_drivers"] == ["efa"] and device["netdevs"] == ["eth0"]
    port = device["ports"]["1"]
    assert port["state"] == "4: ACTIVE" and port["phys_state"] == "5: LinkUp"
    assert port["hw_counters"] == {"rx_drops": "7", "tx_pkts": "100"}
    assert port["counters"] == {"port_rcv_errors": "0"}
    assert value["physical_netdevs"] == ["ens5", "eth0", "eth1"], (
        "a netdev without a device (veth) is not physical"
    )
    assert value["ethtool"]["path"] == "/usr/sbin/ethtool"
    statistics = value["ethtool"]["statistics"]
    assert set(statistics) == {"eth0", "eth1", "ens5"}, (
        "ethtool is listed for the EFA-owned and the physical netdevs"
    )
    assert statistics["eth0"]["counters"] == ["queue_0_tx_cnt", "rnr_nak_err"], (
        "names only, the header line dropped"
    )
    assert all(command[1:2] == ["-S"] for command in seen), seen
    assert not (tmp_path / "run").exists(), "the inventory writes nothing"


def test_inventory_without_ethtool_or_devices_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe.shutil, "which", lambda tool, path=None: None)
    code, value = _main(
        "inventory",
        "--run-id",
        RUN,
        "--source",
        str(tmp_path / "absent"),
        "--netdev-root",
        str(tmp_path / "absent-net"),
        capsys=capsys,
    )
    assert code == 0 and value["devices"] == {} and value["physical_netdevs"] == []
    assert value["ethtool"] == {"path": None, "statistics": {}}


def test_read_source_refuses_paths_outside_the_allow_list(
    host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    for path in (
        "/etc/passwd",
        "/sys/class/infiniband/../../../etc/passwd",
        "sys/class/infiniband/rdmap0s0/ports/1/state",
        "/proc/driver/nvidia/../../etc/shadow",
    ):
        code, value = _main(
            "read-source", "--run-id", RUN, "--path", path, capsys=capsys
        )
        assert code == 1 and value["error"] == "ProbeError", path
    assert probe.checked_source_path("/proc/meminfo") == "/proc/meminfo"
    assert (
        probe.checked_source_path("/proc/driver/nvidia/gpus/0000:53:00.0/information")
        == "/proc/driver/nvidia/gpus/0000:53:00.0/information"
    ), "a file of a tree destination is readable too"


def test_read_source_distinguishes_a_bound_fixture_from_the_real_file(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = b"4: ACTIVE\n"
    _install("state", fixture, capsys)
    dest = "/sys/class/infiniband/rdmap0s0/ports/1/state"
    code, opened = _main(
        "open-window",
        "--run-id",
        RUN,
        "--mode",
        "bind",
        "--bind",
        f"state:{dest}",
        "--skip-verify",
        capsys=capsys,
    )
    assert code == 0, opened
    code, value = _main("read-source", "--run-id", RUN, "--path", dest, capsys=capsys)
    assert code == 0, value
    assert value["window_open"] is True
    assert value["unit"]["readable"] is True and value["unit"]["value"] == "4: ACTIVE"
    assert value["unit"]["device"] is not None
    assert value["host"]["readable"] is False, (
        "PID 1 reads the real sysfs file, which this test host does not have"
    )
    assert value["same"] is False
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True
    code, value = _main("read-source", "--run-id", RUN, "--path", dest, capsys=capsys)
    assert code == 0 and value["window_open"] is False, value
    assert value["unit"]["readable"] is False, (
        "after the close the unit reads the real path again (absent on this test host)"
    )


def test_netdev_mutation_attaches_a_physical_netdev_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    source = _sysfs_infiniband(tmp_path / "host", devices=1)
    net_root = _physical_netdevs(tmp_path / "host", "ens5")
    code, error = _main(
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--netdev",
        "rdmap0s0:veth0",
        "--netdev-root",
        str(net_root),
        capsys=capsys,
    )
    assert code == 1 and error["error"] == "ProbeError", "a veth has no device"
    code, built = _main(
        "build-infiniband-fixture",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--netdev",
        "rdmap0s0:ens5",
        "--netdev-root",
        str(net_root),
        capsys=capsys,
    )
    assert code == 0, built
    tree = probe.fixtures_dir(RUN) / "infiniband"
    assert (tree / "rdmap0s0" / "device" / "net" / "ens5").is_dir(), (
        "the fixture tree attaches the netdev"
    )
    assert built["devices"]["rdmap0s0"]["netdevs"] == ["ens5", "eth0"]
    assert "rdmap0s0/device/net/ens5 (attached)" in built["mutations"]
    assert not (net_root / "ens5" / "net").exists(), "the real netdev is untouched"
