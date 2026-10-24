#!/usr/bin/env python3
"""Network half of the GF-GENERAL-LIVE host probe: host-local test interfaces.

Runs on the target node through ``host_probe_fixture.py`` (chroot into the
host, the node's own Python), standard library only, one JSON object on the
last stdout line.

It only ever creates and drives *test* interfaces named ``gfrs<digest><slot>``
(``gfrs`` + 6 hex of the run id digest + one letter), declared required to the
host collector by a ``collector_window_probe.py`` window the runner opens
separately. It refuses to touch any interface that carries an address or has a
``device`` link in sysfs -- the management ENI (``enp71s0``), the VPC-CNI ENI
(``enp72s0``), pod veths and the EFA netdevs are all excluded by that rule -- so
a real NIC can never be taken down. A ``veth`` pair carries a drop generator (a
raw-socket sender of oversized frames behind a small-MTU peer), a ``dummy``
interface reads ``operstate=unknown``, and a netns peer carries a TCP flow with
``tc netem`` loss for the TCP-only negative. Every interface is created with a
``systemd-run --on-active`` deadman that deletes it, and the before/after link
and address digest of every non-test interface is recorded so an untouched
management plane is provable.
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


def interface_addresses(name: str) -> list[str]:
    completed = run(["ip", "-j", "-o", "addr", "show", "dev", name], check=False)
    try:
        documents = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return sorted(
        f"{addr.get('local')}/{addr.get('prefixlen')}"
        for item in documents
        for addr in item.get("addr_info", [])
        if addr.get("local")
    )


def real_interfaces(run_id: str) -> list[str]:
    result = []
    for entry in sorted(NET_CLASS.iterdir()):
        if entry.name == "lo" or is_test_interface(run_id, entry.name):
            continue
        result.append(entry.name)
    return result


def management_digest(run_id: str) -> dict[str, Any]:
    """A digest of every non-test interface's link and addresses.

    The before/after comparison proves the management plane was untouched.
    """

    snapshot = {
        name: {"link": interface_link(name), "addresses": interface_addresses(name)}
        for name in real_interfaces(run_id)
    }
    routes = run(["ip", "-j", "route", "show"], check=False).stdout
    canonical = json.dumps(
        {"interfaces": snapshot, "routes": routes}, sort_keys=True, default=str
    )
    return {
        "interfaces": sorted(snapshot),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def forbid_real(run_id: str, name: str) -> None:
    if not is_test_interface(run_id, name):
        raise ProbeError("refusing to operate on a non-test interface")
    link = interface_link(name)
    if link.get("present") and (link.get("has_device") or interface_addresses(name)):
        raise ProbeError("test interface name collides with a real device")


EMPTY_STATE: dict[str, Any] = {"interfaces": [], "netns": False, "generators": []}


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


def create_veth(arguments: argparse.Namespace) -> None:
    """A ``veth`` pair ``<a>``/``<b>``, both up; the peer MTU is small."""

    run_id = checked_run_id(arguments.run_id)
    a = interface_name(run_id, arguments.slot)
    b = interface_name(run_id, arguments.peer_slot)
    seconds = bounded(
        arguments.deadman_seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman"
    )
    forbid_real(run_id, a)
    forbid_real(run_id, b)
    before = management_digest(run_id)
    deadman = arm_deadman(run_id, "interfaces", seconds)
    state = load_state(run_id)
    for name in (a, b):
        if name not in state["interfaces"]:
            state["interfaces"].append(name)
    save_state(run_id, state)
    run(["ip", "link", "add", a, "type", "veth", "peer", "name", b])
    run(["ip", "link", "set", b, "mtu", str(OVERSIZE_MTU)])
    run(["ip", "link", "set", a, "up"])
    run(["ip", "link", "set", b, "up"])
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


def residue(run_id: str) -> dict[str, Any]:
    digest = run_digest(run_id)
    live = [
        entry.name
        for entry in NET_CLASS.iterdir()
        if entry.name.startswith(f"{TEST_PREFIX}{digest}")
    ]
    generators = [
        line.split()[0]
        for line in run(
            [
                "systemctl",
                "list-units",
                "--all",
                "--no-legend",
                "--plain",
                f"{UNIT_PREFIX}{digest}-drops*",
            ],
            check=False,
        ).stdout.splitlines()
        if line.strip() and not line.split()[0].endswith(f"{DEADMAN_PREFIX}.timer")
    ]
    netns = run(["ip", "netns", "list"], check=False).stdout
    return {
        "interfaces": sorted(live),
        "generators": [item for item in generators if ".service" in item],
        "netns": [
            line.split()[0]
            for line in netns.splitlines()
            if line.startswith(netns_name(run_id))
        ],
    }


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not value["interfaces"] and not value["generators"] and not value["netns"]


def cleanup(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    kind = arguments.kind
    if kind not in {"all", "interfaces", "drops", "netns"}:
        raise ProbeError("unsafe cleanup kind")
    digest = run_digest(run_id)
    if kind in {"all", "drops"}:
        run(["systemctl", "stop", f"{UNIT_PREFIX}{digest}-drops.service"], check=False)
        run(
            ["systemctl", "reset-failed", f"{UNIT_PREFIX}{digest}-drops.service"],
            check=False,
        )
    if kind in {"all", "netns"}:
        if (
            run(["ip", "netns", "list"], check=False).stdout.find(netns_name(run_id))
            >= 0
        ):
            run(["ip", "netns", "del", netns_name(run_id)], check=False)
    if kind in {"all", "interfaces"}:
        for entry in list(NET_CLASS.iterdir()):
            if entry.name.startswith(f"{TEST_PREFIX}{digest}"):
                run(["ip", "link", "del", entry.name], check=False)
        save_state(run_id, {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))})
    disarm_deadman(run_id, "interfaces")
    remaining = residue(run_id)
    emit(
        {
            "run_id": run_id,
            "kind": kind,
            "residue": remaining,
            "clean": residue_is_clean(remaining),
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
