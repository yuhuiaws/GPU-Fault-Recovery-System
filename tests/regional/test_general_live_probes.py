"""Pure-function contract tests for the GF-GENERAL-LIVE host probes.

The probes run on the node through the host-probe fixture; nothing here runs
systemd, kubectl, losetup or ip. These cover the stdlib-only validators, the
oracle math and the state/residue bookkeeping that decide what the probes may
touch and whether a run left the node clean.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

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


# --------------------------------------------------------------------------- #
# net probe: management-plane digest (GF-GENERAL-LIVE-008/009 create-veth gate)
#
# The before/after digest in create_veth proves creating a gfrs test interface
# left the management plane untouched. On a live node the management ENI carries
# traffic (the probe's own kubectl-exec stream, the node control plane), so the
# digest must ignore the per-interface packet/error/drop counters -- hashing
# them made create_veth raise ProbeError on ordinary traffic before any product
# evidence (the 2026-10-24 live finding). It must still catch a real change.
# --------------------------------------------------------------------------- #
def _mgmt_link(**over: Any) -> dict[str, Any]:
    link: dict[str, Any] = {
        "present": True,
        "operstate": "up",
        "address": "02:aa:bb:cc:dd:ee",
        "mtu": "9001",
        "carrier": "1",
        "ifindex": "2",
        "flags": "0x11043",
        "has_device": True,
        "counters": {
            "rx_errors": 0,
            "tx_errors": 0,
            "rx_dropped": 0,
            "tx_dropped": 0,
            "rx_packets": 0,
            "tx_packets": 0,
        },
    }
    link.update(over)
    return link


def test_net_management_digest_ignores_traffic_and_pod_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "gl8-link-a1-deadbeef"
    moved = {"n": 0}

    def interface_link(name: str) -> dict[str, Any]:
        if name == "enp71s0":  # a real management interface carrying traffic
            moved["n"] += 1
            return _mgmt_link(
                counters={
                    "rx_errors": 0,
                    "tx_errors": 0,
                    "rx_dropped": 0,
                    "tx_dropped": 0,
                    "rx_packets": 1000 * moved["n"],
                    "tx_packets": 900 * moved["n"],
                }
            )
        return _mgmt_link(has_device=False, address=None, ifindex="57")  # pod veth

    monkeypatch.setattr(netp, "real_interfaces", lambda rid: ["cali9f0", "enp71s0"])
    monkeypatch.setattr(netp, "interface_link", interface_link)
    monkeypatch.setattr(
        netp,
        "interface_addresses",
        lambda name, **_kw: ["10.0.0.5/24"] if name == "enp71s0" else [],
    )

    first = netp.management_digest(run_id)
    second = netp.management_digest(run_id)
    assert first["sha256"] == second["sha256"]
    # Only the addressed/device-backed interface is part of the plane; the pod
    # veth (no address, no device link) is excluded so its churn never trips it.
    assert first["interfaces"] == ["enp71s0"]


def test_net_management_digest_flags_a_real_config_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "gl8-link-a1-deadbeef"
    monkeypatch.setattr(netp, "real_interfaces", lambda rid: ["enp71s0"])
    monkeypatch.setattr(
        netp, "interface_addresses", lambda name, **_kw: ["10.0.0.5/24"]
    )

    monkeypatch.setattr(netp, "interface_link", lambda name: _mgmt_link())
    healthy = netp.management_digest(run_id)["sha256"]
    # The same interface taken down must change the digest: that real change is
    # exactly what the before/after comparison exists to catch.
    monkeypatch.setattr(
        netp, "interface_link", lambda name: _mgmt_link(operstate="down", carrier="0")
    )
    assert netp.management_digest(run_id)["sha256"] != healthy
    # A re-addressed interface (new hardware address) is caught too.
    monkeypatch.setattr(
        netp, "interface_link", lambda name: _mgmt_link(address="02:ff:ff:ff:ff:ff")
    )
    assert netp.management_digest(run_id)["sha256"] != healthy


def test_net_create_veth_survives_live_counter_movement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(netp, "STATE_ROOT", tmp_path / "state")
    run_id = "gl8-link-a1-deadbeef"
    a = netp.interface_name(run_id, "a")
    b = netp.interface_name(run_id, "b")
    reads = {"n": 0}

    def interface_link(name: str) -> dict[str, Any]:
        if name in (a, b):
            return {"present": False}
        reads["n"] += 1  # the ENI counters step up between the two digest reads
        return _mgmt_link(
            counters={
                "rx_errors": 0,
                "tx_errors": 0,
                "rx_dropped": 0,
                "tx_dropped": 0,
                "rx_packets": 1000 * reads["n"],
                "tx_packets": 900 * reads["n"],
            }
        )

    monkeypatch.setattr(netp, "real_interfaces", lambda rid: ["enp71s0"])
    monkeypatch.setattr(netp, "interface_link", interface_link)
    monkeypatch.setattr(
        netp,
        "interface_addresses",
        lambda name, **_kw: ["10.0.0.5/24"] if name not in (a, b) else [],
    )
    monkeypatch.setattr(netp, "arm_deadman", lambda rid, kind, seconds: "gfrs-deadman")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr(netp, "run", fake_run)
    emitted: dict[str, Any] = {}
    monkeypatch.setattr(netp, "emit", lambda value: emitted.update(value))

    args = argparse.Namespace(
        run_id=run_id, slot="a", peer_slot="b", deadman_seconds=1680
    )
    netp.create_veth(args)  # must not raise although the ENI counters moved
    assert emitted["a"] == a
    assert emitted["b"] == b
    assert ["ip", "link", "add", a, "type", "veth", "peer", "name", b] in commands


def test_net_create_veth_still_refuses_a_real_interface_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(netp, "STATE_ROOT", tmp_path / "state")
    run_id = "gl8-link-a1-deadbeef"
    a = netp.interface_name(run_id, "a")
    b = netp.interface_name(run_id, "b")
    reads = {"n": 0}

    def interface_link(name: str) -> dict[str, Any]:
        if name in (a, b):
            return {"present": False}
        reads["n"] += 1  # up in the "before" read, down by the "after" read
        return _mgmt_link(operstate="up" if reads["n"] == 1 else "down")

    monkeypatch.setattr(netp, "real_interfaces", lambda rid: ["enp71s0"])
    monkeypatch.setattr(netp, "interface_link", interface_link)
    monkeypatch.setattr(
        netp,
        "interface_addresses",
        lambda name, **_kw: ["10.0.0.5/24"] if name not in (a, b) else [],
    )
    monkeypatch.setattr(netp, "arm_deadman", lambda rid, kind, seconds: "gfrs-deadman")
    monkeypatch.setattr(
        netp,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="[]", stderr=""
        ),
    )
    monkeypatch.setattr(netp, "emit", lambda value: None)

    args = argparse.Namespace(
        run_id=run_id, slot="a", peer_slot="b", deadman_seconds=1680
    )
    with pytest.raises(netp.ProbeError, match="changed a real interface digest"):
        netp.create_veth(args)


# --------------------------------------------------------------------------- #
# net probe: a kernel link-local is not a real-device address
#
# Bringing a veth up auto-assigns an IPv6 link-local (fe80::, scope link). The
# residual 008 ProbeError after the digest fix was forbid_real (via set-link
# down) treating the test veth's own link-local as a colliding real device, and
# management_plane letting link-local-only pod veths back into the digest.
# --------------------------------------------------------------------------- #
def _addr_show(pairs: list[tuple[str, str]]) -> str:
    """An `ip -j -o addr show` document for the given (scope, local) pairs."""
    return json.dumps(
        [
            {
                "addr_info": [
                    {
                        "family": "inet6" if ":" in local else "inet",
                        "local": local,
                        "prefixlen": 64 if ":" in local else 24,
                        "scope": scope,
                    }
                    for scope, local in pairs
                ]
            }
        ]
    )


def _run_returning(addresses: dict[str, list[tuple[str, str]]]) -> Any:
    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["ip", "-j", "-o"] and "addr" in command:
            name = command[-1]
            return subprocess.CompletedProcess(
                command, 0, stdout=_addr_show(addresses.get(name, [])), stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    return runner


def test_net_forbid_real_ignores_the_veths_own_link_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "gl8-link-a1-deadbeef"
    name = netp.interface_name(run_id, "a")
    monkeypatch.setattr(
        netp, "interface_link", lambda n: {"present": True, "has_device": False}
    )
    # An up veth carries only the kernel IPv6 link-local -> not a real device.
    monkeypatch.setattr(netp, "run", _run_returning({name: [("link", "fe80::a")]}))
    netp.forbid_real(run_id, name)  # must not raise
    # A routable (global) address still marks a real device and is refused.
    monkeypatch.setattr(
        netp,
        "run",
        _run_returning({name: [("link", "fe80::a"), ("global", "10.1.2.3")]}),
    )
    with pytest.raises(netp.ProbeError, match="collides with a real device"):
        netp.forbid_real(run_id, name)


def test_net_set_link_down_does_not_refuse_its_own_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "gl8-link-a1-deadbeef"
    c = netp.interface_name(run_id, "c")
    monkeypatch.setattr(
        netp,
        "interface_link",
        lambda n: {"present": True, "has_device": False, "operstate": "up"},
    )
    monkeypatch.setattr(netp, "run", _run_returning({c: [("link", "fe80::c")]}))
    emitted: dict[str, Any] = {}
    monkeypatch.setattr(netp, "emit", lambda value: emitted.update(value))
    netp.set_link(argparse.Namespace(run_id=run_id, slot="c", state="down"))
    assert emitted["name"] == c
    assert emitted["state"] == "down"


def test_net_management_plane_excludes_link_local_only_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "gl8-link-a1-deadbeef"
    monkeypatch.setattr(netp, "real_interfaces", lambda rid: ["enica11b0ad", "enp71s0"])
    monkeypatch.setattr(
        netp, "interface_link", lambda n: _mgmt_link(has_device=(n == "enp71s0"))
    )
    monkeypatch.setattr(
        netp,
        "run",
        _run_returning(
            {
                "enp71s0": [("global", "10.0.0.5")],
                "enica11b0ad": [("link", "fe80::1")],  # pod veth: link-local only
            }
        ),
    )
    # Only the routably-addressed management interface is in the plane; the pod
    # veth (link-local only, no device link) is excluded so its churn is ignored.
    assert sorted(netp.management_plane(run_id)) == ["enp71s0"]


# --------------------------------------------------------------------------- #
# node probe: EFA traffic oracle (GF-GENERAL-LIVE-011)
# --------------------------------------------------------------------------- #
def _efa_port(
    root: Path, device: str, port: str, counters: dict[str, int], *, efa: bool = True
) -> None:
    device_dir = root / device / "device"
    device_dir.mkdir(parents=True, exist_ok=True)
    (device_dir / "uevent").write_text(f"DRIVER={'efa' if efa else 'other'}\n")
    hw = root / device / "ports" / port / "hw_counters"
    hw.mkdir(parents=True, exist_ok=True)
    for name, value in counters.items():
        (hw / name).write_text(str(value))


def test_efa_traffic_reading_sums_the_pair(tmp_path: Path) -> None:
    _efa_port(tmp_path, "rdmap0", "1", {"rx_bytes": 100, "tx_bytes": 40, "rx_pkts": 7})
    _efa_port(tmp_path, "rdmap1", "1", {"rx_bytes": 1000, "tx_bytes": 600})
    reading = nodep.efa_traffic_reading(tmp_path)
    assert reading["rx_bytes_total"] == 1100
    assert reading["tx_bytes_total"] == 640
    assert reading["pair_bytes_total"] == 1740
    assert reading["port_count"] == 2 and reading["complete_pairs"] is True


def test_efa_traffic_reading_ignores_non_efa_and_marks_incomplete(
    tmp_path: Path,
) -> None:
    _efa_port(tmp_path, "mlx0", "1", {"rx_bytes": 9, "tx_bytes": 9}, efa=False)
    _efa_port(tmp_path, "rdmap0", "1", {"only_one": 5})  # no complete pair
    reading = nodep.efa_traffic_reading(tmp_path)
    assert reading["pair_bytes_total"] == 0
    assert reading["port_count"] == 1 and reading["complete_pairs"] is False


def test_efa_traffic_reading_falls_back_to_send_recv(tmp_path: Path) -> None:
    _efa_port(tmp_path, "rdmap0", "1", {"send_bytes": 300, "recv_bytes": 200})
    reading = nodep.efa_traffic_reading(tmp_path)
    assert reading["pair_bytes_total"] == 500 and reading["complete_pairs"] is True


# --------------------------------------------------------------------------- #
# node probe: own-cgroup OOM mode of memory-start, memory-status
# --------------------------------------------------------------------------- #
RUN = "gl2-cgroup-only-a1-deadbeef"
GIB = 1024**3


def _node_probe_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    total_gib: int = 2000,
    available_gib: int = 1200,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    calls: list[list[str]] = []
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(nodep, "STATE_ROOT", tmp_path / "state")

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(nodep, "run", fake_run)
    monkeypatch.setattr(
        nodep, "arm_deadman", lambda run_id, kind, seconds, *a: f"deadman-{kind}"
    )
    monkeypatch.setattr(
        nodep,
        "meminfo",
        lambda: {"MemTotal": total_gib * GIB, "MemAvailable": available_gib * GIB},
    )
    monkeypatch.setattr(nodep, "emit", emitted.append)
    return calls, emitted


def _memory_start(*argv: str) -> None:
    arguments = nodep.parser().parse_args(["memory-start", "--run-id", RUN, *argv])
    arguments.handler(arguments)


def test_node_memory_start_cgroup_oom_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, emitted = _node_probe_runner(monkeypatch, tmp_path)
    _memory_start("--cgroup-oom-bytes", str(2 * GIB), "--seconds", "300", "--approved")
    unit = next(call for call in calls if call[0] == "systemd-run")
    assert "--collect" not in unit, "the failed unit is kept for memory-status"
    assert f"--property=MemoryMax={2 * GIB}" in unit
    assert "--property=MemorySwapMax=0" in unit and "--property=OOMPolicy=kill" in unit
    assert "--property=RuntimeMaxSec=300" in unit
    assert nodep.CGROUP_OOM_ALLOCATOR in unit and nodep.MEMORY_ALLOCATOR not in unit
    assert unit[unit.index(nodep.CGROUP_OOM_ALLOCATOR) + 2] == "300", (
        "argv[2] = seconds"
    )
    assert emitted[-1]["cgroup_oom"] is True
    assert emitted[-1]["memory_max_bytes"] == 2 * GIB
    assert emitted[-1]["target_used_percent"] is None
    assert emitted[-1]["deadman"] == "deadman-memory"


def test_node_memory_start_host_mode_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, emitted = _node_probe_runner(monkeypatch, tmp_path)
    _memory_start("--target-used-percent", "96.5", "--seconds", "300", "--approved")
    unit = next(call for call in calls if call[0] == "systemd-run")
    assert "--collect" in unit
    assert nodep.MEMORY_ALLOCATOR in unit and nodep.CGROUP_OOM_ALLOCATOR not in unit
    assert emitted[-1]["cgroup_oom"] is False
    assert emitted[-1]["target_used_percent"] == 96.5
    assert emitted[-1]["memory_max_bytes"] == int(2000 * GIB * 97.5 / 100)


def test_node_memory_start_cgroup_oom_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _node_probe_runner(monkeypatch, tmp_path)
    with pytest.raises(nodep.ProbeError, match="outside"):
        _memory_start(
            "--cgroup-oom-bytes", str(GIB // 1024), "--seconds", "300", "--approved"
        )
    with pytest.raises(nodep.ProbeError, match="memory target"):
        _memory_start("--seconds", "300", "--approved")
    with pytest.raises(nodep.ProbeError, match="approval"):
        _memory_start("--cgroup-oom-bytes", str(2 * GIB), "--seconds", "300")
    # 2 GiB of a 100 GiB host is 2 %: more than the 1 % the mode allows.
    _node_probe_runner(monkeypatch, tmp_path, total_gib=100, available_gib=60)
    with pytest.raises(nodep.ProbeError, match="1 % of MemTotal or host headroom"):
        _memory_start(
            "--cgroup-oom-bytes", str(2 * GIB), "--seconds", "300", "--approved"
        )
    # A host already at 92.5 % used has no headroom under the 95 % rule.
    _node_probe_runner(monkeypatch, tmp_path, total_gib=2000, available_gib=150)
    with pytest.raises(nodep.ProbeError, match="host headroom"):
        _memory_start(
            "--cgroup-oom-bytes", str(2 * GIB), "--seconds", "300", "--approved"
        )


def test_node_memory_status_reads_the_unit_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, emitted = _node_probe_runner(monkeypatch, tmp_path)
    seen: list[str] = []

    def fake_unit_state(unit: str) -> dict[str, str]:
        seen.append(unit)
        return {"ActiveState": "failed", "Result": "oom-kill", "ExecMainStatus": "9"}

    monkeypatch.setattr(nodep, "unit_state", fake_unit_state)
    arguments = nodep.parser().parse_args(["memory-status", "--run-id", RUN])
    arguments.handler(arguments)
    assert seen == [f"{nodep.unit_name(RUN, 'memory')}.service"]
    assert emitted[-1]["unit"]["Result"] == "oom-kill"
    assert "memory_used_percent" in emitted[-1]["memory"]


def test_node_unit_state_includes_the_oom_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            "LoadState=loaded\nActiveState=failed\nResult=oom-kill\nExecMainStatus=9\n",
            "",
        )

    monkeypatch.setattr(nodep, "run", fake_run)
    state = nodep.unit_state("gfrs-x-memory.service")
    assert state["Result"] == "oom-kill" and state["ExecMainStatus"] == "9"
    assert state["ActiveState"] == "failed" and state["returncode"] == "0"
    properties = [item for item in commands[0] if item.startswith("--property=")]
    assert len(properties) == 1
    assert {"Result", "ExecMainCode", "ExecMainStatus", "ActiveState"} <= set(
        properties[0].removeprefix("--property=").split(",")
    )


# --------------------------------------------------------------------------- #
# node probe: the 002 hold fits the memory unit's cap; failures name their step
# (live 2026-10-25: memory_pressure asked for 1980 s, the cap was 1800 s)
# --------------------------------------------------------------------------- #
def _node_shaped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # A p5en.48xlarge as the two live surveys read it: 2 TiB, 3.6 % used, no swap.
    return _node_probe_runner(monkeypatch, tmp_path, total_gib=2000, available_gib=1928)


def test_node_memory_start_accepts_the_002_hold_on_a_2tib_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls, emitted = _node_shaped(monkeypatch, tmp_path)
    _memory_start("--target-used-percent", "96.5", "--seconds", "1980", "--approved")
    unit = next(call for call in calls if call[0] == "systemd-run")
    assert "--property=RuntimeMaxSec=1980" in unit
    assert f"--property=MemoryMax={int(2000 * GIB * 97.5 / 100)}" in unit
    assert emitted[-1]["deadman"] == "deadman-memory"
    assert nodep.MAX_SECONDS == 3600, "the memory unit shares the load units' cap"
    with pytest.raises(nodep.ProbeError, match="outside 60..3600"):
        _memory_start(
            "--target-used-percent", "96.5", "--seconds", "3601", "--approved"
        )


def test_node_probe_failure_json_names_the_failing_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, emitted = _node_shaped(monkeypatch, tmp_path)
    code = nodep.main(
        [
            "memory-start",
            "--run-id",
            RUN,
            "--target-used-percent",
            "96.5",
            "--seconds",
            "3601",
            "--approved",
        ]
    )
    assert code == 1
    failure = emitted[-1]
    assert failure["error_kind"] == "probe" and failure["error_class"] == "ProbeError"
    assert failure["command"] == "memory-start"
    assert failure["error_site"].startswith("memory_start:"), failure["error_site"]
    assert len(failure["error_code"]) == 16 and int(failure["error_code"], 16) >= 0
    assert "3601" not in json.dumps(failure), "the message itself stays on the node"


def test_node_exception_site_names_the_step_not_the_helper() -> None:
    # memory_status validates its run id through the checked_run_id helper: the
    # site the fixture may show is the step (memory_status), not the helper.
    arguments = nodep.parser().parse_args(["memory-status", "--run-id", "bad id; rm"])
    try:
        arguments.handler(arguments)
    except nodep.ProbeError as exc:
        site = nodep.exception_site(exc)
    assert site is not None and site.startswith("memory_status:"), site
    assert nodep.exception_site(ValueError("no traceback")) is None


def test_node_unit_pause_survives_a_product_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pause outlives the product's own restart of the unit.

    The Node Agent's TRIGGER_HEALTH_SNAPSHOT step restarts the metrics and host
    collectors; a bare SIGSTOP ends with the old process (live 2026-10-25: the
    inventory channel of the paused metrics unit tripped that refresh 183 s in,
    and the new process kept GPU_METRICS fresh). The pause installs a /run
    drop-in that stops every new main process, loaded before the SIGSTOP, and
    the resume removes it before SIGCONT; a leftover is residue.
    """

    calls, emitted = _node_probe_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(nodep, "PAUSE_DROPIN_ROOT", tmp_path / "run-systemd")
    monkeypatch.setattr(
        nodep, "unit_state", lambda unit: {"ActiveState": "active", "MainPID": "7"}
    )
    monkeypatch.setattr(nodep, "unit_is_stopped", lambda unit: True)
    monkeypatch.setattr(nodep, "disarm_deadman", lambda run_id, kind: None)
    monkeypatch.setattr(nodep.shutil, "which", lambda name: "/usr/bin/kill")
    unit = "gpu-fault-metrics-collector.service"
    arguments = nodep.parser().parse_args(
        ["unit-pause", "--run-id", RUN, "--unit", unit, "--seconds", "840"]
    )
    arguments.handler(arguments)

    dropin = nodep.pause_dropin_path(RUN, unit)
    assert dropin.is_file(), "the pause drop-in is installed"
    assert "ExecStartPost=/usr/bin/kill -STOP $MAINPID" in dropin.read_text(), (
        dropin.read_text()
    )
    reload_at = calls.index(["systemctl", "daemon-reload"])
    stop_at = next(i for i, c in enumerate(calls) if "--signal=SIGSTOP" in c)
    assert reload_at < stop_at, calls
    assert emitted[-1]["stopped"] is True, emitted[-1]
    assert emitted[-1]["dropin"] == str(dropin), emitted[-1]
    held = nodep.residue(RUN)
    assert held["pause_dropins"] == [str(dropin)], held
    assert not nodep.residue_is_clean(held), held

    resumed = nodep.resume_unit(RUN, unit)
    assert not dropin.exists(), "the resume removes the drop-in"
    reloads = [i for i, c in enumerate(calls) if c == ["systemctl", "daemon-reload"]]
    cont_at = next(i for i, c in enumerate(calls) if "--signal=SIGCONT" in c)
    assert len(reloads) == 2 and reloads[-1] < cont_at, calls
    assert resumed["dropin"]["removed"] is True, resumed
    cleared = nodep.residue(RUN)
    assert cleared["pause_dropins"] == [] and cleared["paused_units"] == {}, cleared
    assert nodep.residue_is_clean(cleared), cleared
    assert not nodep.residue_is_clean({**cleared, "pause_hooks": [unit]}), (
        "a loaded pause hook is residue"
    )
