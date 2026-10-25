#!/usr/bin/env python3
"""Network half of the GF-GENERAL-LIVE host probe: host-local test interfaces.

Runs on the target node through ``host_probe_fixture.py`` (chroot into the
host, the node's own Python), standard library only, one JSON object on the
last stdout line.

It only ever creates and drives *test* interfaces named ``gfrs<digest><slot>``
(``gfrs`` + 6 hex of the run id digest + one letter), declared required to the
host collector by a ``collector_window_probe.py`` window the runner opens
separately. It only ever operates on ``gfrs*`` test names, and even then refuses
any that carries a routable (non-link-local) address or has a ``device`` link in
sysfs -- the management ENI (``enp71s0``), the VPC-CNI ENI (``enp72s0``) and the
EFA netdevs can never be taken down. The IPv6 link-local (``fe80::``) the kernel
auto-assigns every interface as it comes up -- the probe's own test ``veth`` ends
and host-side pod veths included -- is not a real-device address. A ``veth`` pair
carries a drop generator (a
raw-socket sender of oversized frames behind a small-MTU peer), a ``dummy``
interface reads ``operstate=unknown``, and a netns peer carries a TCP flow with
``tc netem`` loss for the TCP-only negative. Every interface is created with a
``systemd-run --on-active`` deadman that deletes it, and the before/after
configuration digest of every real (addressed or device-backed) non-test
interface is recorded so an untouched management plane is provable.

The TCP-only negative (GF-GENERAL-LIVE-009 ``tcp_only``) is ``tcp-setup`` /
``tcp-start``: a ``gfrs<digest>ns`` test netns with a ``gfrs`` veth pair to the
host netns (host end ``a``, peer ``b`` created inside the netns), static IPv6
link-local test addresses (``scope link``: no routable address and no route on
any other interface), ``tc netem loss`` on the host end's egress only, and a
bounded TCP flow from a sender in the *host* netns to a receiver in the test
netns, so the retransmissions are counted in the host ``/proc/net/snmp`` the
product's collector reads. A named netns is a bind mount under ``/run/netns``
and this probe's ``/host`` is not propagated, so every netns command runs in
PID 1's mount namespace (``nsenter``), where the transient units and the
deadman see it. ``tc``/``sch_netem`` themselves come from the node, or for one
run from ``general_live_deps_probe.py`` (``TEMP_TEST_PACKAGE_INSTALL``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-acceptance/general")
NET_CLASS = Path("/sys/class/net")
UNIT_PREFIX = "gfrs-"
DEADMAN_PREFIX = "ndeadman"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
TEST_PREFIX = "gfrs"
SLOT_PATTERN = re.compile(r"^[a-d]$")
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
MAX_SECONDS = 1800
OVERSIZE_MTU = 576
FRAME_BYTES = 1214
#: Interface-link fields that move on their own on a live node and so must never
#: enter the management-plane digest. Packet/error/drop counters step up on
#: every active real interface between any two reads -- the management ENI alone
#: carries the probe's own ``kubectl exec`` stream and all of the node's
#: control-plane traffic -- so hashing them made the before/after equality in
#: ``create_veth`` fail on ordinary traffic rather than on a real change (the
#: 2026-10-24 live finding: 008/009 ``ProbeError`` before any product evidence).
#: They are still reported verbatim for the *test* interfaces.
VOLATILE_LINK_FIELDS = ("counters",)
#: The qdisc module the TCP fixture needs loaded (or built in).
NETEM_MODULE = "sch_netem"
SYS_MODULE = Path("/sys/module")
MODULES_ROOT = Path("/lib/modules")
PROC_NET_SNMP = Path("/proc/net/snmp")
#: The TCP fixture: static IPv6 link-local test addresses on the gfrs pair
#: (``scope link`` -- never a routable address, never a host route).
HOST_ADDRESS = "fe80::1"
PEER_ADDRESS = "fe80::2"
TCP_PORT = 47913
MAX_LOSS_PERCENT = 20
MAX_RATE_KBPS = 100_000
FLOW_KINDS = ("tcpsend", "tcprecv")


class ProbeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def checked_run_id(value: str) -> str:
    if SAFE_RUN_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def bounded(value: int, low: int, high: int, label: str) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ProbeError(f"{label} is outside {low}..{high}")
    return number


def run_digest(run_id: str) -> str:
    return hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:6]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / hashlib.sha256(run_id.encode()).hexdigest()[:12]


def test_prefix(run_id: str) -> str:
    return f"{TEST_PREFIX}{run_digest(run_id)}"


def interface_name(run_id: str, slot: str) -> str:
    if SLOT_PATTERN.fullmatch(slot) is None:
        raise ProbeError("interface slot must be a..d")
    return f"{test_prefix(run_id)}{slot}"


def netns_name(run_id: str) -> str:
    return f"{test_prefix(run_id)}ns"


def is_test_interface(run_id: str, name: str) -> bool:
    return name.startswith(test_prefix(run_id))


def checked_test_interface(run_id: str, name: str) -> str:
    prefix = test_prefix(run_id)
    if (
        not name.startswith(prefix)
        or SLOT_PATTERN.fullmatch(name[len(prefix) :]) is None
    ):
        raise ProbeError("not a test interface of this run")
    return name


def probe_copy(run_id: str) -> Path:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "net-probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        ["systemctl", "show", unit, "--property=ActiveState", "--property=MainPID"],
        check=False,
    )
    values = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def arm_deadman(run_id: str, kind: str, seconds: int) -> str:
    bounded(seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman seconds")
    name = f"{UNIT_PREFIX}{run_digest(run_id)}-{DEADMAN_PREFIX}-{kind}"
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={int(seconds)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            sys.executable,
            str(probe_copy(run_id)),
            "cleanup",
            "--run-id",
            run_id,
            "--kind",
            kind,
        ]
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return name


def disarm_deadman(run_id: str, kind: str) -> None:
    name = f"{UNIT_PREFIX}{run_digest(run_id)}-{DEADMAN_PREFIX}-{kind}"
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


def interface_link(name: str) -> dict[str, Any]:
    root = NET_CLASS / name
    if not root.exists():
        return {"present": False}
    info: dict[str, Any] = {"present": True}
    for field in ("operstate", "address", "mtu", "carrier", "ifindex", "flags"):
        try:
            info[field] = (root / field).read_text().strip()
        except OSError:
            info[field] = None
    info["has_device"] = (root / "device").exists()
    try:
        info["counters"] = {
            counter: int((root / "statistics" / counter).read_text().strip())
            for counter in (
                "rx_errors",
                "tx_errors",
                "rx_dropped",
                "tx_dropped",
                "rx_packets",
                "tx_packets",
            )
        }
    except (OSError, ValueError):
        info["counters"] = None
    return info


def interface_addresses(name: str, *, include_link_local: bool = True) -> list[str]:
    """Addresses assigned to ``name``.

    ``include_link_local=False`` drops the kernel-assigned link-local addresses
    (``scope link``: IPv6 ``fe80::/10``, IPv4 ``169.254/16``) that appear on
    every interface the moment it is brought up -- the probe's own test ``veth``
    ends and every host-side pod veth included -- so they are never mistaken for
    a real device's address.
    """

    completed = run(["ip", "-j", "-o", "addr", "show", "dev", name], check=False)
    try:
        documents = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return sorted(
        f"{addr.get('local')}/{addr.get('prefixlen')}"
        for item in documents
        for addr in item.get("addr_info", [])
        if addr.get("local") and (include_link_local or addr.get("scope") != "link")
    )


def real_interfaces(run_id: str) -> list[str]:
    result = []
    for entry in sorted(NET_CLASS.iterdir()):
        if entry.name == "lo" or is_test_interface(run_id, entry.name):
            continue
        result.append(entry.name)
    return result


def management_plane(run_id: str) -> dict[str, Any]:
    """The stable configuration of every interface the probe must not touch.

    The management plane is exactly the set ``forbid_real`` protects: a non-test
    interface that carries a routable (non-link-local) address or a sysfs
    ``device`` link (the management ENI, the VPC-CNI ENI and the EFA netdevs).
    Ephemeral pod veths -- only a kernel link-local and no ``device`` link,
    created and torn down as Pods churn -- are not part of it and are left out,
    as are ``VOLATILE_LINK_FIELDS`` and the per-Pod CNI
    route table: the digest proves the management plane's *configuration* is
    untouched (no real interface downed, re-addressed or renumbered by creating
    a test interface), not that no packet moved on it.
    """

    plane: dict[str, Any] = {}
    for name in real_interfaces(run_id):
        link = interface_link(name)
        addresses = interface_addresses(name, include_link_local=False)
        if not (link.get("has_device") or addresses):
            continue
        plane[name] = {
            "link": {
                field: value
                for field, value in link.items()
                if field not in VOLATILE_LINK_FIELDS
            },
            "addresses": addresses,
        }
    return plane


def management_digest(run_id: str) -> dict[str, Any]:
    """A digest of the management plane's stable configuration.

    The before/after comparison proves creating a test interface left every
    real (addressed or device-backed) interface untouched. See
    ``management_plane`` for what is -- and deliberately is not -- hashed.
    """

    plane = management_plane(run_id)
    canonical = json.dumps(plane, sort_keys=True, default=str)
    return {
        "interfaces": sorted(plane),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def forbid_real(run_id: str, name: str) -> None:
    if not is_test_interface(run_id, name):
        raise ProbeError("refusing to operate on a non-test interface")
    link = interface_link(name)
    # A routable address or a sysfs ``device`` link marks a pre-existing real
    # device. The IPv6 link-local the kernel assigns the test veth the moment it
    # comes up must not count, or ``set-link down`` on our own interface would
    # refuse itself (the 2026-10-24 residual 008 ProbeError after the digest fix).
    if link.get("present") and (
        link.get("has_device") or interface_addresses(name, include_link_local=False)
    ):
        raise ProbeError("test interface name collides with a real device")


EMPTY_STATE: dict[str, Any] = {
    "interfaces": [],
    "netns": False,
    "generators": [],
    "flows": [],
    "tcp": None,
}


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "net-state.json"
    if not path.is_file():
        return {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("net probe state identity mismatch")
    for key, empty in EMPTY_STATE.items():
        value.setdefault(key, json.loads(json.dumps(empty)))
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "net-state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def tool_path(name: str) -> str | None:
    return shutil.which(name)


def tool_present(name: str) -> bool:
    return tool_path(name) is not None


def host_mount_ns(command: list[str]) -> list[str]:
    """``command`` in PID 1's mount namespace.

    A named netns is a bind mount under ``/run/netns``. This probe's ``/host``
    is not propagated (the network-namespace analogue of the storage probe's
    mount-propagation finding), so a netns added from the chroot would be an
    empty file to the host's systemd: the ``ip netns exec`` receiver unit and
    the deadman could not enter or delete it. The probe Pod is hostPID, so
    PID 1 is the host's init.
    """

    return ["nsenter", "--target", "1", "--mount", "--", *command]


def add_test_veth(
    run_id: str,
    a: str,
    b: str,
    *,
    deadman_kind: str,
    seconds: int,
    peer_mtu: int | None = None,
    peer_netns: str | None = None,
) -> dict[str, Any]:
    """Create the gfrs veth pair ``a``/``b`` behind the real-interface guards.

    Both names are refused unless they are this run's test names that collide
    with no real device; the management-plane digest is taken first and the
    deadman armed and the state recorded before anything is created. The peer
    stays in the host netns (``peer_mtu`` optionally shrinks it, the drop
    generator's trick) or is created directly inside ``peer_netns``, a test
    netns added here in PID 1's mount namespace. The caller compares the digest
    once its own changes are made.
    """

    forbid_real(run_id, a)
    forbid_real(run_id, b)
    before = management_digest(run_id)
    deadman = arm_deadman(run_id, deadman_kind, seconds)
    state = load_state(run_id)
    for name in (a, b):
        if name not in state["interfaces"]:
            state["interfaces"].append(name)
    if peer_netns is not None:
        state["netns"] = True
    save_state(run_id, state)
    if peer_netns is None:
        run(["ip", "link", "add", a, "type", "veth", "peer", "name", b])
        if peer_mtu is not None:
            run(["ip", "link", "set", b, "mtu", str(peer_mtu)])
        run(["ip", "link", "set", a, "up"])
        run(["ip", "link", "set", b, "up"])
    else:
        run(host_mount_ns(["ip", "netns", "add", peer_netns]))
        run(
            host_mount_ns(
                ["ip", "link", "add", a, "type", "veth"]
                + ["peer", "name", b, "netns", peer_netns]
            )
        )
        run(["ip", "link", "set", a, "up"])
        run(host_mount_ns(["ip", "-n", peer_netns, "link", "set", b, "up"]))
    return {"before": before, "deadman": deadman}


def create_veth(arguments: argparse.Namespace) -> None:
    """A ``veth`` pair ``<a>``/``<b>``, both up; the peer MTU is small."""

    run_id = checked_run_id(arguments.run_id)
    a = interface_name(run_id, arguments.slot)
    b = interface_name(run_id, arguments.peer_slot)
    seconds = bounded(
        arguments.deadman_seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman"
    )
    created = add_test_veth(
        run_id,
        a,
        b,
        deadman_kind="interfaces",
        seconds=seconds,
        peer_mtu=OVERSIZE_MTU,
    )
    before, deadman = created["before"], created["deadman"]
    after = management_digest(run_id)
    if after["sha256"] != before["sha256"]:
        raise ProbeError("creating the test veth changed a real interface digest")
    emit(
        {
            "a": a,
            "b": b,
            "deadman": deadman,
            "link_a": interface_link(a),
            "link_b": interface_link(b),
            "management_digest": before["sha256"],
        }
    )


def create_dummy(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = interface_name(run_id, arguments.slot)
    seconds = bounded(
        arguments.deadman_seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman"
    )
    forbid_real(run_id, name)
    deadman = arm_deadman(run_id, "interfaces", seconds)
    state = load_state(run_id)
    if name not in state["interfaces"]:
        state["interfaces"].append(name)
    save_state(run_id, state)
    run(["ip", "link", "add", name, "type", "dummy"])
    run(["ip", "link", "set", name, "up"])
    emit({"name": name, "deadman": deadman, "link": interface_link(name)})


def set_link(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_test_interface(run_id, interface_name(run_id, arguments.slot))
    if arguments.state not in {"up", "down"}:
        raise ProbeError("link state must be up or down")
    forbid_real(run_id, name) if arguments.state == "down" else None
    run(["ip", "link", "set", name, arguments.state])
    emit({"name": name, "state": arguments.state, "link": interface_link(name)})


def delete_interface(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    name = checked_test_interface(run_id, interface_name(run_id, arguments.slot))
    run(["ip", "link", "del", name], check=False)
    state = load_state(run_id)
    if name in state["interfaces"]:
        state["interfaces"].remove(name)
    save_state(run_id, state)
    emit({"deleted": name, "present": interface_link(name).get("present", False)})


DROP_GENERATOR = r"""
import socket, sys, time
interface, frame_bytes, rate, seconds = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
sock.bind((interface, 0))
payload = b"\xff" * frame_bytes
deadline = time.monotonic() + seconds
interval = 1.0 / rate if rate > 0 else 0.01
while time.monotonic() < deadline:
    try:
        sock.send(payload)
    except OSError:
        pass
    time.sleep(interval)
"""


def start_drops(arguments: argparse.Namespace) -> None:
    """Send oversized frames on ``<a>`` so both ends count ``*_dropped`` deltas."""

    run_id = checked_run_id(arguments.run_id)
    name = checked_test_interface(run_id, interface_name(run_id, arguments.slot))
    seconds = bounded(arguments.seconds, 10, MAX_SECONDS, "seconds")
    rate = bounded(arguments.rate, 1, 1000, "rate")
    unit = f"{UNIT_PREFIX}{run_digest(run_id)}-drops"
    run(["systemctl", "stop", f"{unit}.service"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={unit}",
            "--collect",
            f"--property=RuntimeMaxSec={seconds}",
            "--property=Nice=19",
            sys.executable,
            "-c",
            DROP_GENERATOR,
            name,
            str(FRAME_BYTES),
            str(rate),
            str(seconds),
        ]
    )
    state = load_state(run_id)
    if unit not in state["generators"]:
        state["generators"].append(unit)
    save_state(run_id, state)
    emit(
        {
            "unit": unit,
            "interface": name,
            "seconds": seconds,
            "state": unit_state(f"{unit}.service"),
        }
    )


def stop_drops(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    unit = f"{UNIT_PREFIX}{run_digest(run_id)}-drops"
    run(["systemctl", "stop", f"{unit}.service"], check=False)
    run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    state = load_state(run_id)
    if unit in state["generators"]:
        state["generators"].remove(unit)
    save_state(run_id, state)
    emit({"stopped": unit})


# --------------------------------------------------------------------------- #
# The TCP-only fixture: test netns, gfrs veth pair, host-side netem, TCP flow
# --------------------------------------------------------------------------- #
def module_ready(name: str) -> bool:
    """``name`` is loaded, or built into the running kernel."""

    if (SYS_MODULE / name).exists():
        return True
    try:
        listing = (MODULES_ROOT / os.uname().release / "modules.builtin").read_text()
    except OSError:
        return False
    return any(line.strip().endswith(f"/{name}.ko") for line in listing.splitlines())


def flow_unit(run_id: str, kind: str) -> str:
    if kind not in FLOW_KINDS:
        raise ProbeError("unknown flow unit")
    return f"{UNIT_PREFIX}{run_digest(run_id)}-{kind}"


def netns_names() -> list[str]:
    command = ["ip", "netns", "list"]
    if tool_present("nsenter"):
        command = host_mount_ns(command)
    listing = run(command, check=False).stdout
    return [line.split()[0] for line in listing.splitlines() if line.strip()]


def tcp_retransmits() -> int | None:
    """Host-netns TCP ``RetransSegs`` (the probe Pod is hostNetwork)."""

    try:
        lines = PROC_NET_SNMP.read_text().splitlines()
    except OSError:
        return None
    for header, values in zip(lines, lines[1:]):
        if header.startswith("Tcp:") and values.startswith("Tcp:"):
            names = header.split()[1:]
            numbers = values.split()[1:]
            if "RetransSegs" in names:
                return int(numbers[names.index("RetransSegs")])
    return None


def netem_qdisc(run_id: str, name: str, loss_percent: int) -> None:
    """``tc netem loss`` as the root (egress) qdisc of one gfrs test interface."""

    checked_test_interface(run_id, name)
    forbid_real(run_id, name)
    run(
        ["tc", "qdisc", "add", "dev", name, "root", "netem", "loss", f"{loss_percent}%"]
    )


def qdisc_reading(name: str) -> dict[str, Any]:
    """The root qdisc of ``name`` and its counters (``tc -s qdisc show``)."""

    completed = run(["tc", "-s", "qdisc", "show", "dev", name], check=False)
    reading: dict[str, Any] = {"kind": None, "root": False, "drops": None}
    current: dict[str, Any] | None = None
    for line in completed.stdout.splitlines():
        tokens = line.split()
        if tokens[:1] == ["qdisc"] and len(tokens) >= 4:
            current = {"kind": tokens[1], "root": tokens[3] == "root"}
            loss = re.search(r"\bloss (?:random )?([0-9.]+%)", line)
            current["loss"] = loss.group(1) if loss else None
            if current["root"]:
                reading.update(current)
            continue
        sent = re.search(r"Sent (\d+) bytes (\d+) pkt \(dropped (\d+)", line)
        if sent and current is not None and current.get("root"):
            reading["sent_bytes"] = int(sent.group(1))
            reading["sent_packets"] = int(sent.group(2))
            reading["drops"] = int(sent.group(3))
    return reading


def peer_counters(netns: str, peer: str) -> dict[str, Any]:
    """The netns-side veth end's counters, read inside the test netns."""

    completed = run(
        host_mount_ns(["ip", "-n", netns, "-s", "-j", "link", "show", "dev", peer]),
        check=False,
    )
    try:
        stats = (json.loads(completed.stdout or "[]") or [{}])[0].get("stats64") or {}
    except (json.JSONDecodeError, AttributeError, IndexError, KeyError, TypeError):
        return {"error": "unreadable"}
    return {
        f"{direction}_{counter}": (stats.get(direction) or {}).get(counter)
        for direction in ("rx", "tx")
        for counter in ("errors", "dropped", "packets")
    }


def read_status(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def tcp_setup(arguments: argparse.Namespace) -> None:
    """The test netns, the gfrs pair into it, link-local addresses, host netem."""

    run_id = checked_run_id(arguments.run_id)
    seconds = bounded(
        arguments.deadman_seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman"
    )
    loss = bounded(arguments.loss_percent, 1, MAX_LOSS_PERCENT, "netem loss percent")
    a = interface_name(run_id, "a")
    b = interface_name(run_id, "b")
    ns = netns_name(run_id)
    for tool in ("ip", "tc", "nsenter", "systemd-run"):
        if not tool_present(tool):
            raise ProbeError(f"the TCP fixture needs {tool}")
    if not module_ready(NETEM_MODULE):
        raise ProbeError(f"{NETEM_MODULE} is not loaded")
    left = residue(run_id)
    if left["interfaces"] or left["netns"] or left["flows"]:
        emit(
            {
                "refused": {
                    "reason": "FIXTURE_NOT_READY",
                    "detail": "an earlier attempt of this run left the test "
                    "interfaces/netns/flow; cleanup --kind tcp first",
                },
                "residue": left,
            }
        )
        return
    created = add_test_veth(
        run_id, a, b, deadman_kind="tcp", seconds=seconds, peer_netns=ns
    )
    state = load_state(run_id)
    state["tcp"] = {"interface": a, "peer": b, "netns": ns, "loss_percent": loss}
    save_state(run_id, state)
    addresses = [
        ["ip", "-6", "addr", "add", f"{HOST_ADDRESS}/64", "dev", a, "nodad"],
        host_mount_ns(
            ["ip", "-n", ns, "-6", "addr", "add", f"{PEER_ADDRESS}/64"]
            + ["dev", b, "nodad"]
        ),
    ]
    for command in addresses:
        done = run(command, check=False)
        if done.returncode:
            emit(
                {
                    "refused": {
                        "reason": "FIXTURE_NOT_READY",
                        "detail": "cannot assign the IPv6 link-local test address "
                        f"(IPv6 disabled?): {done.stderr.strip()[:160]}",
                    },
                    "interface": a,
                }
            )
            return
    netem_qdisc(run_id, a, loss)
    after = management_digest(run_id)
    if after["sha256"] != created["before"]["sha256"]:
        raise ProbeError("creating the TCP fixture changed a real interface digest")
    emit(
        {
            "interface": a,
            "peer": b,
            "netns": ns,
            "deadman": created["deadman"],
            "addresses": {a: HOST_ADDRESS, b: PEER_ADDRESS},
            "port": TCP_PORT,
            "qdisc": qdisc_reading(a),
            "link": interface_link(a),
            "real_interfaces": after["interfaces"],
            "management_digest": after["sha256"],
            "refused": None,
        }
    )


TCP_RECEIVER = r"""
import json, os, socket, sys, time
interface, address, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
seconds, status = float(sys.argv[4]), sys.argv[5]
deadline = time.monotonic() + seconds
counts = {"accepted": 0, "received_bytes": 0, "errors": 0}
written = [0.0]
def report(final=False):
    if not final and time.monotonic() - written[0] < 5:
        return
    written[0] = time.monotonic()
    with open(status + ".tmp", "w") as handle:
        json.dump(dict(counts, final=final, at=time.time()), handle)
    os.replace(status + ".tmp", status)
server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((address, port, 0, socket.if_nametoindex(interface)))
server.listen(4)
server.settimeout(1.0)
while time.monotonic() < deadline:
    report()
    try:
        connection, _ = server.accept()
    except socket.timeout:
        continue
    except OSError:
        counts["errors"] += 1
        time.sleep(1)
        continue
    counts["accepted"] += 1
    connection.settimeout(1.0)
    try:
        while time.monotonic() < deadline:
            report()
            try:
                data = connection.recv(65536)
            except socket.timeout:
                continue
            if not data:
                break
            counts["received_bytes"] += len(data)
    except OSError:
        counts["errors"] += 1
    finally:
        connection.close()
report(final=True)
"""

TCP_SENDER = r"""
import json, os, socket, sys, time
interface, source, target, port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
seconds, rate, status = float(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
index = socket.if_nametoindex(interface)
chunk = bytes(16384)
interval = len(chunk) * 8.0 / (rate * 1000.0)
deadline = time.monotonic() + seconds
counts = {"connections": 0, "sent_bytes": 0, "errors": 0}
written = [0.0]
def report(final=False):
    if not final and time.monotonic() - written[0] < 5:
        return
    written[0] = time.monotonic()
    with open(status + ".tmp", "w") as handle:
        json.dump(dict(counts, final=final, at=time.time()), handle)
    os.replace(status + ".tmp", status)
while time.monotonic() < deadline:
    report()
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        sock.settimeout(5.0)
        sock.bind((source, 0, 0, index))
        sock.connect((target, port, 0, index))
        counts["connections"] += 1
        due = time.monotonic()
        while time.monotonic() < deadline:
            report()
            sock.sendall(chunk)
            counts["sent_bytes"] += len(chunk)
            due = max(due + interval, time.monotonic() - 1.0)
            pause = due - time.monotonic()
            if pause > 0:
                time.sleep(pause)
    except OSError:
        counts["errors"] += 1
        time.sleep(1)
    finally:
        sock.close()
report(final=True)
"""


def tcp_start(arguments: argparse.Namespace) -> None:
    """A bounded TCP flow: sender in the host netns, receiver in the test netns."""

    run_id = checked_run_id(arguments.run_id)
    seconds = bounded(arguments.seconds, 30, MAX_SECONDS, "flow seconds")
    rate = bounded(arguments.rate_kbps, 100, MAX_RATE_KBPS, "flow rate")
    state = load_state(run_id)
    tcp = state.get("tcp") or {}
    if not tcp:
        raise ProbeError("no TCP fixture; run tcp-setup first")
    a = checked_test_interface(run_id, str(tcp["interface"]))
    b = checked_test_interface(run_id, str(tcp["peer"]))
    ns = str(tcp["netns"])
    if ns != netns_name(run_id) or not interface_link(a).get("present"):
        raise ProbeError("the TCP fixture is gone")
    if ns not in netns_names():
        raise ProbeError("the test netns is gone")
    ip = tool_path("ip")
    if ip is None:
        raise ProbeError("the TCP flow needs ip")
    directory = state_dir(run_id)
    receiver, sender = flow_unit(run_id, "tcprecv"), flow_unit(run_id, "tcpsend")
    for unit in (receiver, sender):
        run(["systemctl", "stop", f"{unit}.service"], check=False)
        run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
        if unit not in state["flows"]:
            state["flows"].append(unit)
    save_state(run_id, state)
    bound = [f"--property=RuntimeMaxSec={seconds + 30}", "--property=Nice=19"]
    run(
        ["systemd-run", f"--unit={receiver}", "--collect", *bound]
        + [ip, "netns", "exec", ns, sys.executable, "-c", TCP_RECEIVER]
        + [b, PEER_ADDRESS, str(TCP_PORT), str(seconds + 15)]
        + [str(directory / "tcp-receiver.json")]
    )
    run(
        ["systemd-run", f"--unit={sender}", "--collect", *bound]
        + [sys.executable, "-c", TCP_SENDER, a, HOST_ADDRESS, PEER_ADDRESS]
        + [str(TCP_PORT), str(seconds), str(rate), str(directory / "tcp-sender.json")]
    )
    emit(
        {
            "units": {
                unit: unit_state(f"{unit}.service") for unit in (receiver, sender)
            },
            "seconds": seconds,
            "rate_kbps": rate,
            "interface": a,
            "netns": ns,
        }
    )


def tcp_reading(arguments: argparse.Namespace) -> None:
    """The oracle's point readings: RetransSegs, netem counters, flow progress."""

    run_id = checked_run_id(arguments.run_id)
    state = load_state(run_id)
    tcp = state.get("tcp") or {}
    a = tcp.get("interface")
    directory = state_dir(run_id)
    emit(
        {
            "t": time.time(),
            "at": utc_now(),
            "retrans_segs": tcp_retransmits(),
            "interface": a,
            "qdisc": qdisc_reading(checked_test_interface(run_id, a)) if a else None,
            "link": interface_link(a) if a else None,
            "peer": peer_counters(str(tcp["netns"]), str(tcp["peer"])) if tcp else None,
            "sender": read_status(directory / "tcp-sender.json"),
            "receiver": read_status(directory / "tcp-receiver.json"),
            "units": {unit: unit_state(f"{unit}.service") for unit in state["flows"]},
        }
    )


def stop_flows(run_id: str) -> None:
    for kind in FLOW_KINDS:
        unit = flow_unit(run_id, kind)
        run(["systemctl", "stop", f"{unit}.service"], check=False)
        run(["systemctl", "reset-failed", f"{unit}.service"], check=False)
    state = load_state(run_id)
    state["flows"] = []
    save_state(run_id, state)


def tcp_stop(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    stop_flows(run_id)
    directory = state_dir(run_id)
    emit(
        {
            "stopped": [flow_unit(run_id, kind) for kind in FLOW_KINDS],
            "sender": read_status(directory / "tcp-sender.json"),
            "receiver": read_status(directory / "tcp-receiver.json"),
        }
    )


def restore_tcp(run_id: str) -> None:
    """Stop the flow and drop the netem qdisc and the test netns (idempotent)."""

    stop_flows(run_id)
    tcp = load_state(run_id).get("tcp") or {}
    interface = tcp.get("interface")
    if not (interface and interface_link(interface).get("present")):
        return
    try:
        forbid_real(run_id, checked_test_interface(run_id, interface))
    except ProbeError:
        return  # never a qdisc change on anything that looks like a real NIC
    if tool_present("tc"):
        run(["tc", "qdisc", "del", "dev", interface, "root"], check=False)


def delete_netns(run_id: str) -> None:
    name = netns_name(run_id)
    if name in netns_names():
        command = ["ip", "netns", "del", name]
        run(host_mount_ns(command) if tool_present("nsenter") else command, check=False)


def reading(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    names = (
        [interface_name(run_id, slot) for slot in arguments.slot]
        if arguments.slot
        else [name for name in load_state(run_id)["interfaces"]]
    )
    emit(
        {
            "interfaces": {name: interface_link(name) for name in names},
            "management_digest": management_digest(run_id),
        }
    )


def run_units(run_id: str, kind: str) -> list[str]:
    """This run's ``gfrs-<digest>-<kind>*`` services still known to systemd."""

    listing = run(
        [
            "systemctl",
            "list-units",
            "--all",
            "--no-legend",
            "--plain",
            f"{UNIT_PREFIX}{run_digest(run_id)}-{kind}*",
        ],
        check=False,
    ).stdout
    return [
        line.split()[0]
        for line in listing.splitlines()
        if line.strip() and ".service" in line.split()[0]
    ]


def residue(run_id: str) -> dict[str, Any]:
    digest = run_digest(run_id)
    live = [
        entry.name
        for entry in NET_CLASS.iterdir()
        if entry.name.startswith(f"{TEST_PREFIX}{digest}")
    ]
    return {
        "interfaces": sorted(live),
        "generators": run_units(run_id, "drops"),
        "flows": run_units(run_id, "tcp"),
        "netns": [
            name for name in netns_names() if name.startswith(netns_name(run_id))
        ],
    }


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not any(
        value.get(key) for key in ("interfaces", "generators", "netns", "flows")
    )


CLEANUP_KINDS = ("all", "interfaces", "drops", "netns", "tcp")


def cleanup(arguments: argparse.Namespace) -> None:
    """Undo this run's test objects; ``tcp`` is the TCP fixture's deadman kind.

    ``tcp`` stops the flow units, drops the netem qdisc, deletes the test
    netns and every ``gfrs<digest>`` interface. Its deadman is only disarmed
    once that left nothing behind, so a cleanup that could not finish is
    retried when the deadman fires.
    """

    run_id = checked_run_id(arguments.run_id)
    kind = arguments.kind
    if kind not in CLEANUP_KINDS:
        raise ProbeError("unsafe cleanup kind")
    digest = run_digest(run_id)
    if kind in {"all", "drops"}:
        run(["systemctl", "stop", f"{UNIT_PREFIX}{digest}-drops.service"], check=False)
        run(
            ["systemctl", "reset-failed", f"{UNIT_PREFIX}{digest}-drops.service"],
            check=False,
        )
    if kind in {"all", "tcp"}:
        restore_tcp(run_id)
    if kind in {"all", "netns", "tcp"}:
        delete_netns(run_id)
    if kind in {"all", "interfaces", "tcp"}:
        for entry in list(NET_CLASS.iterdir()):
            if entry.name.startswith(f"{TEST_PREFIX}{digest}"):
                run(["ip", "link", "del", entry.name], check=False)
        save_state(run_id, {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))})
    disarm_deadman(run_id, "interfaces")
    remaining = residue(run_id)
    clean = residue_is_clean(remaining)
    if kind in {"all", "tcp"} and clean:
        disarm_deadman(run_id, "tcp")
    emit(
        {
            "run_id": run_id,
            "kind": kind,
            "residue": remaining,
            "clean": clean,
        }
    )


def show_residue(arguments: argparse.Namespace) -> None:
    value = residue(checked_run_id(arguments.run_id))
    emit({"residue": value, "clean": residue_is_clean(value)})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Any) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.set_defaults(handler=handler)
        return item

    item = command("create-veth", create_veth)
    item.add_argument("--slot", default="a")
    item.add_argument("--peer-slot", default="b")
    item.add_argument("--deadman-seconds", type=int, required=True)
    item = command("create-dummy", create_dummy)
    item.add_argument("--slot", default="d")
    item.add_argument("--deadman-seconds", type=int, required=True)
    item = command("set-link", set_link)
    item.add_argument("--slot", required=True)
    item.add_argument("--state", required=True)
    command("delete-interface", delete_interface).add_argument("--slot", required=True)
    item = command("start-drops", start_drops)
    item.add_argument("--slot", default="a")
    item.add_argument("--rate", type=int, default=10)
    item.add_argument("--seconds", type=int, required=True)
    command("stop-drops", stop_drops)
    item = command("reading", reading)
    item.add_argument("--slot", action="append", default=[])
    item = command("tcp-setup", tcp_setup)
    item.add_argument("--deadman-seconds", type=int, required=True)
    item.add_argument("--loss-percent", type=int, default=5)
    item = command("tcp-start", tcp_start)
    item.add_argument("--seconds", type=int, required=True)
    item.add_argument("--rate-kbps", type=int, default=8000)
    command("tcp-reading", tcp_reading)
    command("tcp-stop", tcp_stop)
    command("residue", show_residue)
    command("cleanup", cleanup).add_argument("--kind", default="all")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "command": arguments.command,
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
