"""Host source replay probe: the PCI root fixtures of GF-GENERAL-LIVE-013 ``pci_missing``.

The ``build-pci-fixture`` shapes beyond the synthetic root (a copy of the real
root without named EFA functions, an unreadable root), the ``pci-root`` window
over the unreadable file, the ``snapshot`` fields a post-reboot residue check
reads and the ``inventory``'s PCI function per RDMA device, driven the way
``test_general_live_source_probe.py`` drives the probe: over its ``FakeHost``
(systemctl / systemd-run / nsenter modelled, nothing run on this host).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.collectors.host.observation import efa_inventory_counts
from scripts.e2e.regional.probes import general_live_source_builders as builders
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.regional import test_general_live_source_probe as probe_tests

RUN = probe_tests.RUN
FakeHost = probe_tests.FakeHost
host = probe_tests.host


def _main(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = probe.main(list(argv))
    out = capsys.readouterr().out.strip().splitlines()
    return code, json.loads(out[-1])


# --------------------------------------------------------------------------- #
# PCI root fixtures: the real root without one function, an unreadable root (013)
# --------------------------------------------------------------------------- #
def _fake_pci_root(tmp_path: Path) -> Path:
    """``/sys/bus/pci/devices``: BDF symlinks to functions with vendor/device files."""

    devices = tmp_path / "sys" / "devices" / "pci0000:50"
    root = tmp_path / "sys" / "bus" / "pci" / "devices"
    root.mkdir(parents=True)
    for bdf, vendor, device in (
        ("0000:00:00.0", "0x10de", "0x2330"),
        ("0000:00:05.0", "0x1d0f", "0xec20"),
        ("0000:53:00.0", "0x1d0f", "0xefa2"),
        ("0000:53:01.0", "0x1d0f", "0xefa2"),
    ):
        function = devices / bdf
        function.mkdir(parents=True)
        (function / "vendor").write_text(vendor + "\n")
        (function / "device").write_text(device + "\n")
        (function / "config").write_bytes(b"\0" * 64)  # never copied
        (root / bdf).symlink_to(function)
    (root / "not-a-function").mkdir()  # ignored: not a BDF
    return root


def _pci_namespace(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "run_id": RUN,
        "efa_count": None,
        "malformed": False,
        "source": None,
        "drop": [],
        "unreadable": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_pci_fixture_copies_the_real_root_without_the_dropped_function(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _fake_pci_root(tmp_path)
    code, built = _main(
        "build-pci-fixture",
        "--run-id",
        RUN,
        "--source",
        str(root),
        "--drop",
        "0000:53:01.0",
        capsys=capsys,
    )
    assert code == 0, built
    assert (built["efa_devices"], built["dropped"]) == (1, ["0000:53:01.0"]), built
    assert built["efa_functions"] == ["0000:53:00.0"] and built["source"] == str(root)
    assert built["unreadable"] is False and built["entries"] == 3
    fixture = Path(built["path"])
    assert sorted(path.name for path in fixture.iterdir()) == [
        "0000:00:00.0",
        "0000:00:05.0",
        "0000:53:00.0",
    ]
    assert sorted(path.name for path in (fixture / "0000:53:00.0").iterdir()) == [
        "device",
        "vendor",
    ], "only what the product reads is copied"
    assert (fixture / "0000:00:05.0" / "device").read_text() == "0xec20\n"
    empty = tmp_path / "sys" / "class" / "infiniband"
    empty.mkdir(parents=True)
    assert efa_inventory_counts(empty, fixture) == (1, 0, 0), (
        "the product counts one EFA function less in the copy"
    )


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"drop": ["0000:00:05.0"]}, "not an EFA function"),
        ({"drop": ["0000:99:00.0"]}, "names no PCI function of"),
        ({"drop": ["bogus"]}, "names no PCI function"),
        ({"source": None, "efa_count": 2, "unreadable": True}, "exactly one"),
        (
            {"source": None, "drop": ["0000:53:00.0"], "unreadable": True},
            "--drop needs",
        ),
        ({"source": None, "malformed": True, "unreadable": True}, "--malformed needs"),
        ({"source": None}, "exactly one"),
    ],
)
def test_pci_fixture_refuses_what_is_not_one_reviewed_shape(
    host: FakeHost, tmp_path: Path, overrides: dict[str, Any], needle: str
) -> None:
    root = _fake_pci_root(tmp_path)
    arguments = _pci_namespace(**{"source": str(root), **overrides})
    with pytest.raises(probe.ProbeError, match=needle):
        builders.build_pci_fixture(arguments)
    assert not (probe.fixtures_dir(RUN) / "pci").exists(), (
        "a refused build leaves no half-built root"
    )


def test_unreadable_pci_root_is_a_file_the_product_cannot_list(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, built = _main(
        "build-pci-fixture", "--run-id", RUN, "--unreadable", capsys=capsys
    )
    assert code == 0, built
    assert built["unreadable"] is True and built["efa_devices"] == 0, built
    fixture = Path(built["path"])
    assert fixture.is_file(), "the unreadable root is a regular file"
    empty = tmp_path / "ib"
    empty.mkdir()
    with pytest.raises(NotADirectoryError):
        efa_inventory_counts(empty, fixture)
    code, opened = _main(
        "open-window", "--run-id", RUN, "--mode", "pci-root", capsys=capsys
    )
    assert code == 0, opened
    assert host.environment[probe.PCI_ENV] == str(fixture)
    assert opened["verification"]["ok"] is True, opened["verification"]
    code, status = _main("status", "--run-id", RUN, capsys=capsys)
    assert code == 0, status
    assert status["fixtures"]["pci"].startswith("gfrs-live: unreadable"), status
    code, snap = _main("snapshot", "--run-id", RUN, capsys=capsys)
    assert code == 0, snap
    assert snap["unit_pci_root"] == str(fixture) and snap["window_open"] is True
    assert snap["unit_active_state"] == "active", snap
    code, closed = _main("close-window", "--run-id", RUN, capsys=capsys)
    assert code == 0 and closed["clean"] is True, closed
    code, snap = _main("snapshot", "--run-id", RUN, capsys=capsys)
    assert code == 0, snap
    assert snap["unit_pci_root"] is None, snap
    assert not snap["dropin"] and not snap["deadman"] and not snap["window_open"]
    # Building over the file restores a directory fixture.
    code, built = _main(
        "build-pci-fixture", "--run-id", RUN, "--efa-count", "2", capsys=capsys
    )
    assert code == 0, built
    assert Path(built["path"]).is_dir(), built
    assert built["efa_functions"] == ["0000:53:00.0", "0000:53:01.0"], built


def test_inventory_names_the_pci_function_behind_each_device(
    host: FakeHost, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "sys" / "class" / "infiniband"
    function = tmp_path / "sys" / "devices" / "pci0000:50" / "0000:53:00.0"
    function.mkdir(parents=True)
    (function / "uevent").write_text("DRIVER=efa\n")
    device = source / "rdmap0s0"
    (device / "ports" / "1").mkdir(parents=True)
    (device / "ports" / "1" / "state").write_text("4: ACTIVE\n")
    (device / "ports" / "1" / "phys_state").write_text("5: LinkUp\n")
    (device / "device").symlink_to(function)
    other = source / "rdmap1s0"
    (other / "ports" / "1").mkdir(parents=True)
    (other / "device").mkdir()  # no PCI function behind it
    (other / "device" / "uevent").write_text("DRIVER=efa\n")
    net_root = tmp_path / "sys" / "class" / "net"
    net_root.mkdir(parents=True)
    code, listed = _main(
        "inventory",
        "--run-id",
        RUN,
        "--source",
        str(source),
        "--netdev-root",
        str(net_root),
        capsys=capsys,
    )
    assert code == 0, listed
    assert listed["devices"]["rdmap0s0"]["pci"] == "0000:53:00.0", listed["devices"]
    assert listed["devices"]["rdmap1s0"]["pci"] is None, listed["devices"]
    assert listed["devices"]["rdmap0s0"]["efa"] is True
