"""Pure-function contract tests for the GF-GENERAL-LIVE host probes.

The probes run on the node through the host-probe fixture; nothing here runs
systemd, kubectl, losetup or ip. These cover the stdlib-only validators, the
oracle math and the state/residue bookkeeping that decide what the probes may
touch and whether a run left the node clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.e2e.regional.probes import general_live_net_probe as netp
from scripts.e2e.regional.probes import general_live_node_probe as nodep
from scripts.e2e.regional.probes import general_live_storage_probe as storep


# --------------------------------------------------------------------------- #
# node probe
# --------------------------------------------------------------------------- #
def test_node_identifier_and_bounds() -> None:
    assert nodep.checked_run_id("gl1-cpu-a1-deadbeef") == "gl1-cpu-a1-deadbeef"
    with pytest.raises(nodep.ProbeError, match="unsafe run ID"):
        nodep.checked_run_id("bad id; rm")
    assert nodep.checked_name("fs") == "fs"
    with pytest.raises(nodep.ProbeError):
        nodep.checked_name("TOObig")
    assert nodep.bounded(5, 1, 10, "x") == 5
    with pytest.raises(nodep.ProbeError, match="outside"):
        nodep.bounded(0, 1, 10, "x")
    first = nodep.run_digest("gl1-cpu-a1-deadbeef")
    assert first == nodep.run_digest("gl1-cpu-a1-deadbeef") and len(first) == 12
    assert nodep.unit_name("gl1-cpu-a1-deadbeef", "cpu").startswith("gfrs-"), (
        "nodep.unit_name('gl1-cpu-a1-deadbeef', 'cpu').startswith('gfrs-')"
    )


def test_node_memory_and_statvfs(tmp_path: Path) -> None:
    values = {
        "MemTotal": 100.0,
        "MemAvailable": 4.0,
        "Cached": 50.0,
        "SReclaimable": 10.0,
        "Shmem": 5.0,
        "SwapTotal": 0.0,
    }
    pct = nodep.memory_percentages(values)
    assert round(pct["memory_used_percent"]) == 96
    assert round(pct["memory_available_percent"]) == 4
    assert round(pct["page_cache_percent"]) == 55
    reading = nodep.statvfs_reading(str(tmp_path))
    assert reading["read_only"] in (True, False) and reading["used_percent"] is not None
    assert nodep.statvfs_reading(str(tmp_path / "missing"))["error"]


def test_node_oracle_sample_cpu_and_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nodep, "cpu_counters", lambda: (1000.0, 200.0))
    previous: dict = {}
    first = nodep.oracle_sample({"cpu": True}, previous)
    assert "cpu_busy_percent" not in first  # first sample has no delta
    monkeypatch.setattr(nodep, "cpu_counters", lambda: (2000.0, 300.0))
    second = nodep.oracle_sample({"cpu": True}, previous)
    assert round(second["cpu_busy_percent"]) == 90
    monkeypatch.setattr(
        nodep, "diskstats_fields", lambda device: [float(i) for i in range(15)]
    )
    prev2: dict = {}
    nodep.oracle_sample({"diskstats": ["dm-7"]}, prev2)
    monkeypatch.setattr(
        nodep, "diskstats_fields", lambda device: [float(i) + 100 for i in range(15)]
    )
    out = nodep.oracle_sample({"diskstats": ["dm-7"]}, prev2)
    assert "diskstats" in out and out["diskstats"]["dm-7"]["present"]


def test_node_residue_is_clean_and_process_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = {
        "units": [],
        "page_cache_files": [],
        "paused_units": {},
        "paused_processes": {},
        "log_collector_enabled_by_run": False,
    }
    assert nodep.residue_is_clean(clean), "nodep.residue_is_clean(clean)"
    assert not nodep.residue_is_clean({**clean, "paused_units": {"u": {}}}), (
        "not nodep.residue_is_clean({**clean, 'paused_units': {'u': {}}})"
    )
    assert not nodep.residue_is_clean(
        {**clean, "log_collector_enabled_by_run": True}
    ), "not nodep.residue_is_clean({**clean, 'log_collector_enabled_by_run': T"
    assert nodep.checked_pausable("gpu-fault-host-collector.service"), (
        "nodep.checked_pausable('gpu-fault-host-collector.service')"
    )
    with pytest.raises(nodep.ProbeError, match="pause allow-list"):
        nodep.checked_pausable("gpu-fault-node-agent.service")
    assert "gpu-fault-node-agent.service" not in nodep.PAUSABLE_UNITS


def test_node_backing_root_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "var-lib"
    monkeypatch.setattr(nodep, "BACKING_ROOTS", (root,))
    with pytest.raises(nodep.ProbeError, match="not allow-listed"):
        nodep.backing_root(str(tmp_path / "elsewhere"), 1024)


# --------------------------------------------------------------------------- #
# storage probe
# --------------------------------------------------------------------------- #
def test_storage_dm_table_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert storep.dm_table(2048, "/dev/loop0", 0) == "0 2048 linear /dev/loop0 0"
    assert storep.dm_table(2048, "/dev/loop0", 200) == "0 2048 delay /dev/loop0 0 200"
    monkeypatch.setattr(storep, "STATE_ROOT", tmp_path / "state")
    run_id = "gl4-cap-a1-deadbeef"
    state = storep.load_state(run_id)
    assert state["filesystems"] == {} and state["devices"] == {}
    state["filesystems"]["fs"] = {"mountpoint": "/mnt"}
    storep.save_state(run_id, state)
    assert storep.load_state(run_id)["filesystems"]["fs"]["mountpoint"] == "/mnt"
    with pytest.raises(storep.ProbeError, match="no test file system"):
        storep.checked_filesystem(run_id, "zz")


def test_storage_backing_root_and_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "var-lib" / "gpu-fault-acceptance"
    monkeypatch.setattr(storep, "BACKING_ROOTS", (root,))
    with pytest.raises(storep.ProbeError, match="not allow-listed"):
        storep.backing_root(str(tmp_path / "x"), 1024)
    assert storep.residue_is_clean(
        {
            "mounts": [],
            "device_mapper": [],
            "loop_devices": [],
            "backing_files": [],
            "units": [],
        }
    ), "storep.residue_is_clean( { 'mounts': [], 'device_mapper': [], 'loop_de"
    assert not storep.residue_is_clean(
        {
            "mounts": ["/mnt"],
            "device_mapper": [],
            "loop_devices": [],
            "backing_files": [],
            "units": [],
        }
    ), "not storep.residue_is_clean( { 'mounts': ['/mnt'], 'device_mapper': []"


# --------------------------------------------------------------------------- #
# net probe
# --------------------------------------------------------------------------- #
def test_net_interface_names_and_test_guard() -> None:
    run_id = "gl8-link-a1-deadbeef"
    a = netp.interface_name(run_id, "a")
    assert a.startswith("gfrs") and netp.is_test_interface(run_id, a)
    assert netp.checked_test_interface(run_id, a) == a
    with pytest.raises(netp.ProbeError, match="slot must be"):
        netp.interface_name(run_id, "z")
    with pytest.raises(netp.ProbeError, match="not a test interface"):
        netp.checked_test_interface(run_id, "enp71s0")


def test_net_forbid_real_refuses_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "gl8-link-a1-deadbeef"
    name = netp.interface_name(run_id, "a")
    # A test name that resolves to a real device (has an address) is refused.
    monkeypatch.setattr(
        netp, "interface_link", lambda n: {"present": True, "has_device": True}
    )
    monkeypatch.setattr(netp, "interface_addresses", lambda n: ["10.0.0.1/24"])
    with pytest.raises(netp.ProbeError, match="collides with a real device"):
        netp.forbid_real(run_id, name)
    with pytest.raises(netp.ProbeError, match="non-test interface"):
        netp.forbid_real(run_id, "enp71s0")
    # A test name with no device/address is allowed.
    monkeypatch.setattr(netp, "interface_link", lambda n: {"present": False})
    monkeypatch.setattr(netp, "interface_addresses", lambda n: [])
    netp.forbid_real(run_id, name)


def test_net_state_and_residue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(netp, "STATE_ROOT", tmp_path / "state")
    run_id = "gl9-drops-a1-deadbeef"
    state = netp.load_state(run_id)
    assert state["interfaces"] == []
    state["interfaces"].append(netp.interface_name(run_id, "a"))
    netp.save_state(run_id, state)
    assert netp.load_state(run_id)["interfaces"]
    assert netp.residue_is_clean({"interfaces": [], "generators": [], "netns": []}), (
        "netp.residue_is_clean({'interfaces': [], 'generators': [], 'netns': []"
    )
    assert not netp.residue_is_clean(
        {"interfaces": ["x"], "generators": [], "netns": []}
    ), "not netp.residue_is_clean( {'interfaces': ['x'], 'generators': [], 'ne"
