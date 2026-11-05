"""The GF-GENERAL-LIVE net probe's test interfaces driven through ``main(argv)``.

The TCP fixture's fake host (``ip``/``tc``/``nsenter``/systemd over a fake
``/sys/class/net`` with a real, addressed, device-backed ``enp71s0``) answers
from behind the probe's own ``subprocess`` and ``shutil.which``
(``_cov95_general_live_net.NetBoundary``). Each test runs a subcommand the way
the runner does -- ``create-dummy``, ``delete-interface``, ``start-drops``,
``reading``, the ``tcp-*`` steps, every ``cleanup`` kind -- and checks the JSON,
the state record and the host commands, including the refusals that keep the
probe off the management plane.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_net_probe as netp
from tests.regional._cov95_general_live_net import NetBoundary, install_boundary
from tests.regional.test_general_live_tcp_fixture import (
    NS,
    NSENTER,
    RUN_ID,
    A,
    B,
    FakeHost,
)

Cli = Callable[..., tuple[int, dict[str, Any], str]]
DIGEST = netp.run_digest(RUN_ID)
DUMMY = netp.interface_name(RUN_ID, "d")
DROPS = f"gfrs-{DIGEST}-drops"
DEADMAN = f"gfrs-{DIGEST}-{netp.DEADMAN_PREFIX}"


@pytest.fixture
def node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeHost, NetBoundary]:
    return install_boundary(monkeypatch, tmp_path)


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]) -> Cli:
    def invoke(*argv: str) -> tuple[int, dict[str, Any], str]:
        code = netp.main(list(argv))
        captured = capsys.readouterr()
        lines = captured.out.strip().splitlines()
        assert lines, f"the probe prints one JSON object for {argv}: {captured}"
        return code, json.loads(lines[-1]), captured.err

    return invoke


def call(command: str, *extra: str) -> list[str]:
    return [command, "--run-id", RUN_ID, *extra]


def refused(answer: dict[str, Any], command: str) -> bool:
    return answer == {
        "error": "ProbeError",
        "error_kind": "probe",
        "error_class": "ProbeError",
        "command": command,
    }


def tcp_state(**values: Any) -> None:
    state = netp.load_state(RUN_ID)
    state["tcp"] = {"interface": A, "peer": B, "netns": NS, "loss_percent": 5, **values}
    netp.save_state(RUN_ID, state)


def inner(host: FakeHost) -> list[list[str]]:
    return [
        c[len(NSENTER) :] if c[: len(NSENTER)] == NSENTER else c for c in host.calls
    ]


# --------------------------------------------------------------------------- #
# dummy interfaces
# --------------------------------------------------------------------------- #
def test_create_dummy_arms_its_deadman_records_it_and_brings_it_up(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    code, answer, err = cli(*call("create-dummy", "--deadman-seconds", "600"))
    assert code == 0, err
    assert (answer["name"], answer["deadman"]) == (DUMMY, f"{DEADMAN}-interfaces")
    assert answer["link"]["present"] is True and answer["link"]["operstate"] == "up"
    assert netp.load_state(RUN_ID)["interfaces"] == [DUMMY], "recorded before add"
    commands = inner(host)
    started = next(index for index, c in enumerate(commands) if c[0] == "systemd-run")
    added = commands.index(["ip", "link", "add", DUMMY, "type", "dummy"])
    assert started < added, "the deadman is armed before the interface exists"
    assert commands[started][-4:] == ["--run-id", RUN_ID, "--kind", "interfaces"]

    code, answer, err = cli(*call("create-dummy", "--deadman-seconds", "600"))
    assert code == 1 and refused(answer, "create-dummy"), answer
    assert f"command failed (2): ip link add {DUMMY} type dummy" in err, err
    assert "RTNETLINK answers: File exists" in err, "the tool's error is carried"
    assert netp.load_state(RUN_ID)["interfaces"] == [DUMMY], "recorded once"


def test_create_dummy_without_an_armed_deadman_creates_nothing(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, boundary = node
    boundary.timer_active = False
    code, answer, err = cli(*call("create-dummy", "--deadman-seconds", "600"))
    assert code == 1 and refused(answer, "create-dummy"), answer
    assert f"deadman {DEADMAN}-interfaces is not armed" in err, err
    assert not (host.net_class / DUMMY).exists(), "no interface without a deadman"


def test_delete_interface_removes_the_link_and_its_record(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    assert cli(*call("create-dummy", "--deadman-seconds", "600"))[0] == 0, "created"
    for _ in range(2):  # deleting again finds nothing left to forget
        code, answer, _ = cli(*call("delete-interface", "--slot", "d"))
        assert code == 0 and answer == {"deleted": DUMMY, "present": False}, answer
        assert netp.load_state(RUN_ID)["interfaces"] == [], "the record is dropped"
    assert inner(host).count(["ip", "link", "del", DUMMY]) == 2, host.calls


@pytest.mark.parametrize(
    "link",
    [{"device": True}, {"addresses": [("global", "10.0.44.9")]}],
    ids=["device-link", "routable-address"],
)
def test_delete_interface_refuses_a_test_name_that_is_a_real_device(
    node: tuple[FakeHost, NetBoundary], cli: Cli, link: dict[str, Any]
) -> None:
    host, _ = node
    host.add_link(DUMMY, **link)
    state = netp.load_state(RUN_ID)
    state["interfaces"].append(DUMMY)
    netp.save_state(RUN_ID, state)
    code, answer, err = cli(*call("delete-interface", "--slot", "d"))
    assert code == 1 and refused(answer, "delete-interface"), answer
    assert "test interface name collides with a real device" in err, err
    assert not any(c[:3] == ["ip", "link", "del"] for c in inner(host)), host.calls
    assert (host.net_class / DUMMY).is_dir(), "the interface is left in place"
    assert netp.load_state(RUN_ID)["interfaces"] == [DUMMY], "and so is its record"


def test_set_link_refuses_a_state_that_is_not_up_or_down(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    code, answer, err = cli(*call("set-link", "--slot", "a", "--state", "sideways"))
    assert code == 1 and refused(answer, "set-link"), answer
    assert "link state must be up or down" in err, err
    assert host.calls == [], "nothing runs for an unknown state"


def test_the_cli_refuses_unsafe_ids_and_bounds(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    code, answer, err = cli("residue", "--run-id", "BAD ID")
    assert code == 1 and refused(answer, "residue"), answer
    assert "unsafe run ID" in err, err
    code, answer, err = cli(*call("start-drops", "--seconds", "5"))
    assert code == 1 and "seconds is outside 10..1800" in err, err
    assert host.calls == [], "a refused argument runs no command"


# --------------------------------------------------------------------------- #
# the drop generator
# --------------------------------------------------------------------------- #
def test_start_and_stop_the_bounded_drop_generator(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    for _ in range(2):  # a restarted generator is recorded once
        code, answer, _ = cli(*call("start-drops", "--seconds", "60", "--rate", "50"))
        assert code == 0, answer
    assert (answer["unit"], answer["interface"], answer["seconds"]) == (DROPS, A, 60)
    assert answer["state"]["ActiveState"] == "active", answer["state"]
    started = [c for c in inner(host) if c[0] == "systemd-run"][-1]
    assert started[1:5] == [
        f"--unit={DROPS}",
        "--collect",
        "--property=RuntimeMaxSec=60",
        "--property=Nice=19",
    ], "the generator is a bounded, niced transient unit"
    assert started[-5:] == [netp.DROP_GENERATOR, A, str(netp.FRAME_BYTES), "50", "60"]
    assert netp.load_state(RUN_ID)["generators"] == [DROPS], "recorded once"

    for _ in range(2):
        code, answer, _ = cli(*call("stop-drops"))
        assert code == 0 and answer == {"stopped": DROPS}, answer
        assert netp.load_state(RUN_ID)["generators"] == [], "the record is dropped"


# --------------------------------------------------------------------------- #
# readings
# --------------------------------------------------------------------------- #
def test_reading_reports_unreadable_link_fields_as_none(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    assert cli(*call("create-dummy", "--deadman-seconds", "600"))[0] == 0, "created"
    link = host.net_class / DUMMY
    (link / "carrier").unlink()
    (link / "statistics" / "rx_errors").write_text("n/a\n", encoding="utf-8")
    code, answer, _ = cli(*call("reading", "--slot", "d"))
    assert code == 0, answer
    reading = answer["interfaces"][DUMMY]
    assert reading["carrier"] is None and reading["counters"] is None, reading
    assert reading["operstate"] == "up", "the readable fields are still reported"
    assert answer["management_digest"]["interfaces"] == ["enp71s0"], answer

    code, recorded, _ = cli(*call("reading"))
    assert list(recorded["interfaces"]) == [DUMMY], "without slots: the recorded ones"


def test_a_state_file_of_another_run_is_refused(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    path = netp.state_dir(RUN_ID) / "net-state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id": "gl009-other-run"}), encoding="utf-8")
    code, answer, err = cli(*call("reading"))
    assert code == 1 and refused(answer, "reading"), answer
    assert "net probe state identity mismatch" in err, err


def test_tcp_retransmits_reads_only_the_tcp_rows(
    node: tuple[FakeHost, NetBoundary], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snmp = tmp_path / "snmp-variants"
    monkeypatch.setattr(netp, "PROC_NET_SNMP", snmp)
    assert netp.tcp_retransmits() is None, "an unreadable snmp file has no count"
    snmp.write_text("Ip: Forwarding\nIp: 1\nUdp: InDatagrams\nUdp: 5\n")
    assert netp.tcp_retransmits() is None, "no Tcp rows, no count"
    snmp.write_text("Ip: Forwarding\nIp: 1\nTcp: InSegs OutSegs\nTcp: 10 9\n")
    assert netp.tcp_retransmits() is None, "a Tcp row without RetransSegs"
    snmp.write_text("Tcp: RetransSegs InSegs\nTcp: 42 10\n")
    assert netp.tcp_retransmits() == 42, "the RetransSegs column of the Tcp row"


def test_qdisc_reading_ignores_a_qdisc_that_is_not_the_root(
    node: tuple[FakeHost, NetBoundary],
) -> None:
    _, boundary = node
    boundary.answers[("tc", "-s", "qdisc")] = (
        0,
        "qdisc pfifo 10: parent 1:1 limit 1000p\n"
        " Sent 600 bytes 6 pkt (dropped 2, overlimits 0 requeues 0)\n",
    )
    assert netp.qdisc_reading(A) == {"kind": None, "root": False, "drops": None}, (
        "a child qdisc and its counters are not the root's"
    )


def test_module_ready_reads_the_running_kernels_builtin_list(
    node: tuple[FakeHost, NetBoundary], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(netp, "SYS_MODULE", tmp_path / "no-modules")
    builtin = netp.MODULES_ROOT / os.uname().release / "modules.builtin"
    assert netp.module_ready("sch_netem") is False, "neither loaded nor listed"
    builtin.parent.mkdir(parents=True)
    builtin.write_text("kernel/net/sched/sch_tbf.ko\n", encoding="utf-8")
    assert netp.module_ready("sch_netem") is False, "another module is built in"
    builtin.write_text("kernel/net/sched/sch_netem.ko\n", encoding="utf-8")
    assert netp.module_ready("sch_netem") is True, "built into the running kernel"


def test_netns_names_without_nsenter_lists_the_probes_own_namespaces(
    node: tuple[FakeHost, NetBoundary],
) -> None:
    host, boundary = node
    boundary.hidden.add("nsenter")
    assert netp.netns_names() == ["cni-0d1e2f"], "the listing is parsed"
    assert host.calls == [["ip", "netns", "list"]], "run directly, not via nsenter"


def test_flow_unit_refuses_an_unknown_kind() -> None:
    with pytest.raises(netp.ProbeError, match="unknown flow unit"):
        netp.flow_unit(RUN_ID, "udpsend")


def test_interface_driver_of_an_unsearchable_device_is_unknown(
    node: tuple[FakeHost, NetBoundary],
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root searches a mode-0000 directory")
    host, _ = node
    device = host.net_class / "enp71s0" / "device"
    device.chmod(0o000)
    try:
        assert netp.interface_driver("enp71s0") is None, "unreadable: no driver"
    finally:
        device.chmod(0o755)


# --------------------------------------------------------------------------- #
# the TCP fixture's refusals and its stop
# --------------------------------------------------------------------------- #
def test_tcp_setup_needs_every_tool(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, boundary = node
    boundary.hidden.add("tc")
    code, answer, err = cli(*call("tcp-setup", "--deadman-seconds", "600"))
    assert code == 1 and refused(answer, "tcp-setup"), answer
    assert "the TCP fixture needs tc" in err, err
    assert host.calls == [], "nothing is created without tc"


@pytest.mark.parametrize(
    "prepare, message",
    [
        (lambda host, boundary: None, "no TCP fixture; run tcp-setup first"),
        (lambda host, boundary: tcp_state(), "the TCP fixture is gone"),
        (
            lambda host, boundary: (tcp_state(), host.add_link(A)),
            "the test netns is gone",
        ),
        (
            lambda host, boundary: (
                tcp_state(),
                host.add_link(A),
                host.netns.add(NS),
                boundary.hidden.add("ip"),
            ),
            "the TCP flow needs ip",
        ),
    ],
    ids=["no-fixture", "interface-gone", "netns-gone", "no-ip"],
)
def test_tcp_start_refuses_without_a_complete_fixture(
    node: tuple[FakeHost, NetBoundary],
    cli: Cli,
    prepare: Callable[[FakeHost, NetBoundary], object],
    message: str,
) -> None:
    host, boundary = node
    prepare(host, boundary)
    code, answer, err = cli(*call("tcp-start", "--seconds", "60"))
    assert code == 1 and refused(answer, "tcp-start"), answer
    assert message in err, err
    assert not any(c[0] == "systemd-run" for c in inner(host)), "no flow starts"


def test_tcp_start_twice_records_each_flow_unit_once(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    tcp_state()
    host.add_link(A)
    host.netns.add(NS)
    for _ in range(2):
        code, answer, _ = cli(*call("tcp-start", "--seconds", "60"))
        assert code == 0, answer
    assert netp.load_state(RUN_ID)["flows"] == [
        netp.flow_unit(RUN_ID, "tcprecv"),
        netp.flow_unit(RUN_ID, "tcpsend"),
    ], "a restarted flow is recorded once"


def test_tcp_stop_stops_both_flows_and_reads_their_last_status(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    tcp_state()
    directory = netp.state_dir(RUN_ID)
    (directory / "tcp-sender.json").write_text(json.dumps({"sent_bytes": 16384}))
    (directory / "tcp-receiver.json").write_text(json.dumps(["not", "a", "status"]))
    code, answer, _ = cli(*call("tcp-stop"))
    assert code == 0, answer
    assert answer == {
        "stopped": [netp.flow_unit(RUN_ID, kind) for kind in netp.FLOW_KINDS],
        "sender": {"sent_bytes": 16384},
        "receiver": None,
    }, "a status that is not an object is no status"
    assert netp.load_state(RUN_ID)["flows"] == [], "no flow is recorded any more"


def test_tcp_reading_reports_unreadable_peer_counters(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, boundary = node
    tcp_state()
    host.add_link(A)
    boundary.answers[("ip", "-n", NS, "-s", "-j", "link")] = (0, "not json")
    code, answer, _ = cli(*call("tcp-reading"))
    assert code == 0, answer
    assert answer["peer"] == {"error": "unreadable"}, answer["peer"]
    assert answer["retrans_segs"] == 100, "the host netns snmp count"


# --------------------------------------------------------------------------- #
# cleanup kinds
# --------------------------------------------------------------------------- #
def test_each_cleanup_kind_undoes_only_its_own_objects(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    assert cli(*call("create-dummy", "--deadman-seconds", "600"))[0] == 0, "created"
    host.calls.clear()
    code, answer, _ = cli(*call("cleanup", "--kind", "drops"))
    assert code == 0 and answer["clean"] is False, "the dummy is still there"
    assert [c[:2] for c in inner(host) if c[0] == "ip"] == [["ip", "netns"]], (
        "drops touches no interface and no netns (it only lists them)"
    )
    assert ["systemctl", "stop", f"{DROPS}.service"] in host.calls, host.calls

    host.calls.clear()
    code, answer, _ = cli(*call("cleanup", "--kind", "netns"))
    assert not any(c[:3] == ["ip", "netns", "del"] for c in inner(host)), (
        "no test netns exists, none is deleted"
    )

    code, answer, _ = cli(*call("cleanup", "--kind", "interfaces"))
    assert code == 0 and answer["clean"] is True, answer
    assert answer["refused"] == [], "the probe's own dummy is no real device"
    assert ["ip", "link", "del", DUMMY] in inner(host), "the dummy is deleted"
    assert netp.load_state(RUN_ID)["interfaces"] == [], "the state is reset"

    code, answer, err = cli(*call("cleanup", "--kind", "everything"))
    assert code == 1 and "unsafe cleanup kind" in err, err


def test_a_cleanup_never_deletes_a_test_name_that_is_a_real_device(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    assert cli(*call("create-dummy", "--deadman-seconds", "600"))[0] == 0, "created"
    host.add_link(A, device=True)
    host.add_link(B, addresses=[("global", "10.0.44.9")])
    own = netp.interface_name(RUN_ID, "c")
    host.add_link(own, addresses=[("link", "fe80::1")])
    host.calls.clear()
    code, answer, _ = cli(*call("cleanup", "--kind", "interfaces"))
    assert code == 0, answer
    assert answer["refused"] == [A, B], "a device link or a routable address"
    deleted = [c[3] for c in inner(host) if c[:3] == ["ip", "link", "del"]]
    assert deleted == [own, DUMMY], (
        "only the test interfaces are deleted; a kernel link-local is no real address"
    )
    assert (host.net_class / A).is_dir() and (host.net_class / B).is_dir(), (
        "the refused names are left in place"
    )
    assert answer["residue"]["interfaces"] == [A, B], answer["residue"]
    assert answer["clean"] is False, "a cleanup that refused something is not clean"
    assert (host.net_class / "enp71s0").is_dir(), "the management ENI is untouched"


def test_a_tcp_cleanup_without_a_fixture_still_disarms_both_deadmen(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    code, answer, _ = cli(*call("cleanup", "--kind", "tcp"))
    assert code == 0 and answer["clean"] is True, answer
    stopped = [c[2] for c in inner(host) if c[:2] == ["systemctl", "stop"]]
    assert f"{DEADMAN}-tcp.timer" in stopped, "a clean tcp cleanup disarms its deadman"
    assert f"{DEADMAN}-interfaces.timer" in stopped, "and the interfaces deadman"
    assert not any(c[0] == "tc" for c in inner(host)), "no qdisc to drop"


def test_a_tcp_cleanup_never_drops_a_qdisc_on_a_real_device(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    tcp_state()
    host.add_link(A, device=True)  # the recorded name now carries a device link
    code, answer, _ = cli(*call("cleanup", "--kind", "tcp"))
    assert code == 0, answer
    assert not any(c[0] == "tc" for c in inner(host)), (
        "a name that looks like a real NIC gets no qdisc change"
    )
    assert ["ip", "link", "del", A] not in inner(host), "and is never deleted"
    assert (answer["refused"], answer["clean"]) == ([A], False), answer
    stopped = [c[2] for c in inner(host) if c[:2] == ["systemctl", "stop"]]
    assert f"{DEADMAN}-tcp.timer" not in stopped, (
        "the tcp deadman stays armed while the residue is left"
    )


def test_a_tcp_cleanup_without_tc_skips_the_qdisc(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, boundary = node
    tcp_state()
    host.add_link(A)
    boundary.hidden.add("tc")
    code, answer, _ = cli(*call("cleanup", "--kind", "tcp"))
    assert code == 0 and answer["clean"] is True, answer
    assert not any(c[0] == "tc" for c in inner(host)), "no tc, no qdisc command"
    assert ["ip", "link", "del", A] in inner(host), "the test veth still goes"


def test_a_cleanup_run_by_the_deadman_does_not_stop_its_own_service(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, boundary = node
    boundary.deadman_pid = str(os.getpid())
    code, answer, _ = cli(*call("cleanup", "--kind", "interfaces"))
    assert code == 0, answer
    assert ["systemctl", "stop", f"{DEADMAN}-interfaces.timer"] in host.calls
    assert ["systemctl", "stop", f"{DEADMAN}-interfaces.service"] not in host.calls, (
        "the deadman's own cleanup never stops the service it runs in"
    )


def test_residue_lists_this_runs_objects(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, _ = node
    host.add_link(DUMMY)
    host.netns.add(NS)
    code, answer, _ = cli(*call("residue"))
    assert code == 0 and answer["clean"] is False, answer
    assert answer["residue"]["interfaces"] == [DUMMY], answer["residue"]
    assert answer["residue"]["netns"] == [NS], answer["residue"]


# --------------------------------------------------------------------------- #
# veth helpers
# --------------------------------------------------------------------------- #
def test_a_host_side_veth_pair_without_a_peer_mtu_keeps_the_default(
    node: tuple[FakeHost, NetBoundary],
) -> None:
    host, _ = node
    created = netp.add_test_veth(RUN_ID, A, B, deadman_kind="interfaces", seconds=600)
    assert created["deadman"] == f"{DEADMAN}-interfaces", created
    ip = [c for c in inner(host) if c[:3] == ["ip", "link", "set"]]
    assert ip == [["ip", "link", "set", A, "up"], ["ip", "link", "set", B, "up"]], (
        "no MTU change when none is asked for"
    )
    netp.add_test_veth(RUN_ID, "gfrs" + DIGEST + "c", B, deadman_kind="x", seconds=600)
    assert netp.load_state(RUN_ID)["interfaces"] == [A, B, "gfrs" + DIGEST + "c"], (
        "a name already recorded is recorded once"
    )


def test_recreate_veth_refuses_a_change_to_the_management_plane(
    node: tuple[FakeHost, NetBoundary], cli: Cli
) -> None:
    host, boundary = node
    assert cli(*call("create-veth", "--deadman-seconds", "600"))[0] == 0, "created"
    boundary.on_add = lambda: (host.net_class / "enp71s0" / "mtu").write_text("9001\n")
    code, answer, err = cli(*call("recreate-veth", "--deadman-seconds", "600"))
    assert code == 1 and refused(answer, "recreate-veth"), answer
    assert "recreating the test veth changed a real interface digest" in err, err
    assert netp.load_state(RUN_ID)["interfaces"] == [A, B], "recorded once"


def test_running_the_file_as_a_script_exits_with_the_cli_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # An unsafe run ID is refused before the probe reads or writes the host.
    monkeypatch.setattr(sys, "argv", [netp.__file__, "residue", "--run-id", "BAD ID"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(netp.__file__, run_name="__main__")
    assert raised.value.code == 1, "a refused command exits 1"
    captured = capsys.readouterr()
    answer = json.loads(captured.out.strip().splitlines()[-1])
    assert refused(answer, "residue"), answer
    assert "ProbeError: unsafe run ID" in captured.err, captured.err
