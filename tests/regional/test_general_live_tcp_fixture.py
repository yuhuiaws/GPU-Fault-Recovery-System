"""The net probe's TCP-only fixture (GF-GENERAL-LIVE-009 tcp_only).

A recording fake host stands in for ``ip``, ``tc``, ``nsenter`` and systemd
over a fake ``/sys/class/net`` that also holds a real, addressed and
device-backed management interface. The tests prove netem lands only on the
host-side gfrs test veth, the real-interface guards still refuse, the netns
commands run in PID 1's mount namespace, the flow units are bounded, and the
``tcp`` cleanup -- the deadman's command -- removes the flow, qdisc, netns and
veth.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_net_probe as netp

RUN_ID = "gl009-tcp-only-a1-deadbeef"
NSENTER = ["nsenter", "--target", "1", "--mount", "--"]
A = netp.interface_name(RUN_ID, "a")
B = netp.interface_name(RUN_ID, "b")
NS = netp.netns_name(RUN_ID)
SNMP = (
    "Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens "
    "AttemptFails EstabResets CurrEstab InSegs OutSegs RetransSegs InErrs "
    "OutRsts InCsumErrors\n"
    "Tcp: 1 200 120000 -1 10 4 0 0 3 1000 900 {retrans} 0 0 0\n"
)


class FakeHost:
    """``ip``/``tc``/``nsenter``/systemd of one node over a fake sysfs."""

    def __init__(self, root: Path) -> None:
        self.net_class = root / "net"
        self.net_class.mkdir()
        self.calls: list[list[str]] = []
        self.netns: set[str] = {"cni-0d1e2f"}
        self.addresses: dict[str, list[tuple[str, str]]] = {}
        self.qdisc: dict[str, str] = {}
        self.units: set[str] = set()
        self.pairs: dict[str, str] = {}
        self.ipv6 = True
        self.emitted: list[dict[str, Any]] = []
        self.add_link("enp71s0", device=True, addresses=[("global", "10.0.43.5")])

    def add_link(
        self,
        name: str,
        *,
        device: bool = False,
        addresses: Sequence[tuple[str, str]] = (),
    ) -> None:
        root = self.net_class / name
        (root / "statistics").mkdir(parents=True)
        for field, value in (
            ("operstate", "down"),
            ("address", "02:00:00:00:00:01"),
            ("mtu", "1500"),
            ("carrier", "1"),
            ("ifindex", str(len(list(self.net_class.iterdir())) + 1)),
            ("flags", "0x1003"),
        ):
            (root / field).write_text(value + "\n")
        for counter in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped"):
            (root / "statistics" / counter).write_text("0\n")
        for counter in ("rx_packets", "tx_packets"):
            (root / "statistics" / counter).write_text("0\n")
        if device:
            (root / "device").mkdir()
        self.addresses[name] = list(addresses)

    def remove_link(self, name: str) -> None:
        root = self.net_class / name
        if root.exists():
            for item in sorted(root.rglob("*"), reverse=True):
                if item.is_dir():
                    item.rmdir()
                else:
                    item.unlink()
            root.rmdir()

    def ok(self, command: list[str], stdout: str = "", code: int = 0) -> Any:
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")

    def _ip(self, command: list[str], args: list[str]) -> Any:
        if args[:2] == ["netns", "add"]:
            self.netns.add(args[2])
        elif args[:2] == ["netns", "list"]:
            return self.ok(
                command, "".join(f"{n} (id: 1)\n" for n in sorted(self.netns))
            )
        elif args[:2] == ["netns", "del"]:
            self.netns.discard(args[2])
            for host, ns in list(self.pairs.items()):
                if ns == args[2]:
                    self.remove_link(host)
                    del self.pairs[host]
        elif args[:2] == ["link", "add"]:
            self.add_link(args[2])
            if "netns" in args:
                self.pairs[args[2]] = args[args.index("netns") + 1]
        elif args[:2] == ["link", "set"] and args[3] == "up":
            (self.net_class / args[2] / "operstate").write_text("up\n")
        elif args[:2] == ["link", "del"]:
            self.remove_link(args[2])
        elif args[:3] == ["-j", "-o", "addr"]:
            info = [
                {"local": local, "prefixlen": 64, "scope": scope}
                for scope, local in self.addresses.get(args[-1], [])
            ]
            return self.ok(command, json.dumps([{"addr_info": info}]))
        elif args[:3] == ["-6", "addr", "add"]:
            if not self.ipv6:
                return self.ok(command, code=2)
            self.addresses.setdefault(args[5], []).append(("link", args[3][:-3]))
        return self.ok(command)

    def _tc(self, command: list[str], args: list[str]) -> Any:
        if args[:2] == ["qdisc", "add"]:
            self.qdisc[args[3]] = " ".join(args[4:])
        elif args[:2] == ["qdisc", "del"]:
            self.qdisc.pop(args[3], None)
        elif args[:2] == ["-s", "qdisc"]:
            if args[-1] in self.qdisc:
                return self.ok(
                    command,
                    "qdisc netem 8001: root refcnt 2 limit 1000 loss 5%\n"
                    " Sent 151200 bytes 120 pkt (dropped 7, overlimits 0 requeues 0)\n"
                    " backlog 0b 0p requeues 0\n",
                )
            return self.ok(command, "qdisc noqueue 0: root refcnt 2\n")
        return self.ok(command)

    def run(self, command: list[str], **_: Any) -> Any:
        self.calls.append(list(command))
        inner = (
            command[len(NSENTER) :] if command[: len(NSENTER)] == NSENTER else command
        )
        if inner[0] == "ip":
            return self._ip(command, inner[1:])
        if inner[0] == "tc":
            return self._tc(command, inner[1:])
        if inner[0] == "systemd-run":
            self.units.add(next(i for i in inner if i.startswith("--unit="))[7:])
        elif inner[:2] == ["systemctl", "stop"]:
            self.units.discard(inner[2].rsplit(".", 1)[0])
        elif inner[:2] == ["systemctl", "show"]:
            return self.ok(command, "ActiveState=active\nMainPID=0\n")
        elif inner[:2] == ["systemctl", "list-units"]:
            prefix = inner[-1].rstrip("*")
            listed = sorted(u for u in self.units if u.startswith(prefix))
            return self.ok(
                command, "".join(f"{u}.service loaded active\n" for u in listed)
            )
        return self.ok(command)

    def mutations(self) -> list[list[str]]:
        """Every command that changes the host (reads filtered out)."""

        result = []
        for call in self.calls:
            inner = call[len(NSENTER) :] if call[: len(NSENTER)] == NSENTER else call
            if inner[:1] == ["tc"] and inner[1:2] != ["-s"]:
                result.append(inner)
            elif inner[:1] == ["ip"] and not (
                inner[1:3] in (["-j", "-o"], ["netns", "list"]) or "-s" in inner
            ):
                result.append(inner)
            elif inner[:1] == ["systemd-run"]:
                result.append(inner)
        return result


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeHost(tmp_path)
    sys_module = tmp_path / "sys-module"
    (sys_module / "sch_netem").mkdir(parents=True)
    snmp = tmp_path / "snmp"
    snmp.write_text(SNMP.format(retrans=100))
    monkeypatch.setattr(netp, "NET_CLASS", fake.net_class)
    monkeypatch.setattr(netp, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(netp, "SYS_MODULE", sys_module)
    monkeypatch.setattr(netp, "MODULES_ROOT", tmp_path / "lib-modules")
    monkeypatch.setattr(netp, "PROC_NET_SNMP", snmp)
    monkeypatch.setattr(netp, "run", fake.run)
    monkeypatch.setattr(netp, "tool_path", lambda name: f"/usr/sbin/{name}")
    monkeypatch.setattr(netp, "emit", fake.emitted.append)
    return fake


def setup_args(**overrides: Any) -> argparse.Namespace:
    values = {"run_id": RUN_ID, "deadman_seconds": 2400, "loss_percent": 5}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_tcp_setup_puts_netem_on_the_host_side_test_veth_only(host) -> None:
    netp.tcp_setup(setup_args())
    created = host.emitted[-1]
    assert created["refused"] is None, created
    assert (created["interface"], created["peer"], created["netns"]) == (A, B, NS)
    assert created["qdisc"]["kind"] == "netem" and created["qdisc"]["root"] is True
    assert created["real_interfaces"] == ["enp71s0"]
    tc = [call for call in host.mutations() if call[0] == "tc"]
    assert tc == [["tc", "qdisc", "add", "dev", A, "root", "netem", "loss", "5%"]]
    touched = [call for call in host.mutations() if "enp71s0" in call]
    assert touched == [], "no command ever changes the management ENI"
    assert host.addresses["enp71s0"] == [("global", "10.0.43.5")]
    # The peer is created inside the netns, both in PID 1's mount namespace.
    assert [*NSENTER, "ip", "netns", "add", NS] in host.calls
    assert [
        *NSENTER,
        *["ip", "link", "add", A, "type", "veth", "peer", "name", B, "netns", NS],
    ] in host.calls
    assert host.addresses[A] == [("link", "fe80::1")], "link-local test address only"
    assert [
        *NSENTER,
        *["ip", "-n", NS, "-6", "addr", "add", "fe80::2/64", "dev", B, "nodad"],
    ] in host.calls


def test_tcp_setup_arms_the_deadman_before_creating_anything(host) -> None:
    netp.tcp_setup(setup_args())
    mutations = host.mutations()
    assert mutations[0][0] == "systemd-run", mutations[:2]
    deadman = mutations[0]
    assert f"--unit=gfrs-{netp.run_digest(RUN_ID)}-ndeadman-tcp" in deadman
    assert deadman[-5:] == ["cleanup", "--run-id", RUN_ID, "--kind", "tcp"]
    state = netp.load_state(RUN_ID)
    assert state["interfaces"] == [A, B] and state["netns"] is True
    assert state["tcp"] == {"interface": A, "peer": B, "netns": NS, "loss_percent": 5}


def test_netem_never_targets_a_real_interface(host) -> None:
    with pytest.raises(netp.ProbeError, match="not a test interface"):
        netp.netem_qdisc(RUN_ID, "enp71s0", 5)
    host.add_link(A, device=True)
    with pytest.raises(netp.ProbeError, match="collides with a real device"):
        netp.netem_qdisc(RUN_ID, A, 5)
    assert [call for call in host.mutations() if call[0] == "tc"] == []


def test_the_veth_guard_refuses_a_name_that_is_a_real_device(host) -> None:
    host.add_link(B, addresses=[("global", "10.1.2.3")])
    with pytest.raises(netp.ProbeError, match="collides with a real device"):
        netp.add_test_veth(
            RUN_ID, A, B, deadman_kind="tcp", seconds=2400, peer_netns=NS
        )
    assert host.mutations() == [], "refused before any deadman, netns or link"


def test_tcp_setup_refuses_leftovers_of_an_earlier_attempt(host) -> None:
    host.add_link(A)
    netp.tcp_setup(setup_args())
    refused = host.emitted[-1]["refused"]
    assert refused["reason"] == "FIXTURE_NOT_READY", refused
    assert host.mutations() == []


def test_tcp_setup_fails_on_a_management_plane_change(
    host, monkeypatch: pytest.MonkeyPatch
) -> None:
    digests = iter(["before", "after"])
    monkeypatch.setattr(
        netp,
        "management_digest",
        lambda run_id: {"interfaces": ["enp71s0"], "sha256": next(digests)},
    )
    with pytest.raises(netp.ProbeError, match="changed a real interface digest"):
        netp.tcp_setup(setup_args())


def test_tcp_setup_refuses_without_ipv6(host) -> None:
    host.ipv6 = False
    netp.tcp_setup(setup_args())
    refused = host.emitted[-1]["refused"]
    assert refused["reason"] == "FIXTURE_NOT_READY" and "IPv6" in refused["detail"]
    assert [call for call in host.mutations() if call[0] == "tc"] == []


def test_tcp_setup_needs_a_loaded_netem(host, tmp_path: Path) -> None:
    (tmp_path / "sys-module" / "sch_netem").rmdir()
    with pytest.raises(netp.ProbeError, match="sch_netem is not loaded"):
        netp.tcp_setup(setup_args())
    assert host.mutations() == []


def test_tcp_start_runs_a_bounded_flow_from_the_host_netns(host) -> None:
    netp.tcp_setup(setup_args())
    host.calls.clear()
    netp.tcp_start(argparse.Namespace(run_id=RUN_ID, seconds=690, rate_kbps=8000))
    receiver, sender = [call for call in host.calls if call[0] == "systemd-run"]
    digest = netp.run_digest(RUN_ID)
    assert f"--unit=gfrs-{digest}-tcprecv" in receiver
    assert f"--unit=gfrs-{digest}-tcpsend" in sender
    for unit in (receiver, sender):
        assert "--property=RuntimeMaxSec=720" in unit and "--collect" in unit
    start = receiver.index("/usr/sbin/ip")
    assert receiver[start : start + 4] == ["/usr/sbin/ip", "netns", "exec", NS]
    assert receiver[start + 4 : start + 6] == [sys.executable, "-c"]
    assert receiver[start + 7 : start + 10] == [B, "fe80::2", "47913"]
    body = sender.index(sys.executable)
    assert sender[body + 3 : body + 9] == [
        A,
        "fe80::1",
        "fe80::2",
        "47913",
        "690",
        "8000",
    ]
    assert "netns" not in sender, "the sender stays in the host netns"
    assert netp.load_state(RUN_ID)["flows"] == [
        f"gfrs-{digest}-tcprecv",
        f"gfrs-{digest}-tcpsend",
    ]


def test_tcp_reading_reports_host_retransmits_and_netem_drops(host) -> None:
    netp.tcp_setup(setup_args())
    netp.tcp_reading(argparse.Namespace(run_id=RUN_ID))
    reading = host.emitted[-1]
    assert reading["retrans_segs"] == 100
    assert reading["qdisc"] == {
        "kind": "netem",
        "root": True,
        "loss": "5%",
        "sent_bytes": 151200,
        "sent_packets": 120,
        "drops": 7,
    }
    assert reading["link"]["counters"]["tx_dropped"] == 0


def test_tcp_cleanup_removes_flow_qdisc_netns_and_veth(host) -> None:
    netp.tcp_setup(setup_args())
    netp.tcp_start(argparse.Namespace(run_id=RUN_ID, seconds=690, rate_kbps=8000))
    host.calls.clear()
    netp.cleanup(argparse.Namespace(run_id=RUN_ID, kind="tcp"))
    result = host.emitted[-1]
    assert result["clean"] is True, result
    digest = netp.run_digest(RUN_ID)
    assert ["systemctl", "stop", f"gfrs-{digest}-tcpsend.service"] in host.calls
    assert ["systemctl", "stop", f"gfrs-{digest}-tcprecv.service"] in host.calls
    assert ["tc", "qdisc", "del", "dev", A, "root"] in host.calls
    assert [*NSENTER, "ip", "netns", "del", NS] in host.calls
    assert ["systemctl", "stop", f"gfrs-{digest}-ndeadman-tcp.timer"] in host.calls
    assert not (host.net_class / A).exists() and NS not in host.netns
    assert "cni-0d1e2f" in host.netns, "a pod netns is never touched"
    assert (host.net_class / "enp71s0").is_dir(), (
        "(host.net_class / 'enp71s0').is_dir()"
    )
    assert netp.load_state(RUN_ID)["tcp"] is None


def test_tcp_cleanup_keeps_its_deadman_when_residue_is_left(
    host, monkeypatch: pytest.MonkeyPatch
) -> None:
    netp.tcp_setup(setup_args())
    host.calls.clear()
    monkeypatch.setattr(
        netp,
        "residue",
        lambda run_id: {"interfaces": [A], "generators": [], "flows": [], "netns": []},
    )
    netp.cleanup(argparse.Namespace(run_id=RUN_ID, kind="tcp"))
    assert host.emitted[-1]["clean"] is False
    digest = netp.run_digest(RUN_ID)
    assert ["systemctl", "stop", f"gfrs-{digest}-ndeadman-tcp.timer"] not in host.calls


def test_qdisc_reading_reads_only_the_root_qdisc(host) -> None:
    host.qdisc[A] = "root netem loss 5%"
    assert netp.qdisc_reading(A)["drops"] == 7
    assert netp.qdisc_reading("gfrs000000z")["kind"] == "noqueue"
