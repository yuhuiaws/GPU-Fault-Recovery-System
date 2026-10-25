#!/usr/bin/env python3
"""Reversible host-level network isolation for GF-GENERAL-LIVE 028/011.

Runs on a node through ``host_probe_fixture`` (privileged, hostPID,
hostNetwork, chrooted into the host root), stdlib only. Two block kinds, each
a set of tagged ``REJECT --reject-with tcp-reset`` rules with a systemd deadman:

* ``pod-egress`` -- on the node that runs the completion watcher Pod: FORWARD
  rules that reject TCP from that Pod's IP to the control-plane endpoint's
  resolved IPv4 addresses (or the given CIDRs) on its port. Scoped by the
  source Pod IP, so no other Pod, no host process and no other destination is
  touched; the tcp-reset makes the watcher's POSTs fail at once instead of
  hanging to their timeout. GF-GENERAL-LIVE-028 ``api_unreachable``.
* ``agent-ingress`` -- on a GPU node: one INPUT rule that rejects TCP to the
  Node Agent's listening port, the port the cluster executor pushes node
  actions to (``/v1/node-actions/submit``). The Agent's own outbound
  heartbeats, every collector and kubelet are untouched. GF-GENERAL-LIVE-011
  ``unreachable_node``.

Order of every block: the intent is recorded under ``/run`` first, the
``systemd-run --on-active`` deadman is armed (it runs a copy of this file with
``cleanup --strict`` even when the runner, the probe Pod and the control plane
are all gone, and retries until the chain is clean), then the rules are
inserted at the top of their chain. ``cleanup`` deletes every rule carrying the
run's tag, disarms the timer, re-reads the chain and reports what remains; the
runner refuses to call a node clean on anything but an empty remainder.
Nothing here touches a rule without the run's tag.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-acceptance/general")
STATE_FILE = "isolation.json"
UNIT_PREFIX = "gfrs-"
DEADMAN_PREFIX = "ndeadman"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
SAFE_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
#: Block kind -> the filter chain its rules live in. Pod traffic leaves the
#: node through FORWARD (the Pod has its own network namespace); a connection
#: to a host-bound listener such as the Node Agent is delivered through INPUT.
KINDS = {"pod-egress": "FORWARD", "agent-ingress": "INPUT"}
AGENT_UNIT = "gpu-fault-node-agent.service"
#: ``iptables -S`` prints every rule of a chain as ``-A <chain> <spec>``; the
#: run's tag rides in the comment match so nothing else is ever deleted.
APPEND = "-A"
MAX_DELETE_PASSES = 50


class ProbeError(RuntimeError):
    pass


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def run(
    command: list[str], *, check: bool = True, timeout: int = 120
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
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


def checked_kind(value: str) -> str:
    if value not in KINDS:
        raise ProbeError(f"unknown isolation kind {value!r}")
    return value


def checked_port(value: int) -> int:
    return bounded(value, 1, 65535, "port")


def bounded(value: int, low: int, high: int, label: str) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ProbeError(f"{label} is outside {low}..{high}")
    return number


def run_digest(run_id: str) -> str:
    return hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:6]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:12]


def tag(run_id: str, kind: str) -> str:
    """The comment every rule of this run and kind carries (no spaces)."""

    return f"{UNIT_PREFIX}{run_digest(run_id)}-iso-{checked_kind(kind)}"


def deadman_unit(run_id: str, kind: str) -> str:
    return (
        f"{UNIT_PREFIX}{run_digest(run_id)}-{DEADMAN_PREFIX}-iso-{checked_kind(kind)}"
    )


def probe_copy(run_id: str) -> Path:
    """A copy of this file the deadman runs after the probe Pod is gone."""

    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "isolation-probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        [
            "systemctl",
            "show",
            unit,
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=NextElapseUSecRealtime",
        ],
        check=False,
    )
    values: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def arm_deadman(run_id: str, kind: str, seconds: int) -> dict[str, Any]:
    """``systemd-run --on-active``: a strict cleanup of this kind, retried."""

    bounded(seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman seconds")
    name = deadman_unit(run_id, kind)
    timer = unit_state(f"{name}.timer")
    if timer.get("ActiveState") == "active":
        return {"unit": name, "timer": timer, "armed": "already"}
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
            "--strict",
        ]
    )
    timer = unit_state(f"{name}.timer")
    if timer.get("ActiveState") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return {"unit": name, "timer": timer, "armed": "now", "seconds": int(seconds)}


def disarm_deadman(run_id: str, kind: str) -> dict[str, str]:
    name = deadman_unit(run_id, kind)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)
    return unit_state(f"{name}.timer")


def firewall() -> str:
    path = shutil.which("iptables")
    if path is None:
        raise ProbeError("iptables is not installed on the host")
    return path


def rule_spec(
    kind: str,
    rule_tag: str,
    *,
    port: int,
    source: str | None = None,
    destination: str | None = None,
) -> list[str]:
    """The match/target part of one rule (what ``-S`` prints after the chain)."""

    spec: list[str] = []
    if source is not None:
        spec += ["-s", source]
    if destination is not None:
        spec += ["-d", destination]
    spec += [
        "-p",
        "tcp",
        "--dport",
        str(checked_port(port)),
        "-m",
        "comment",
        "--comment",
        rule_tag,
        "-j",
        "REJECT",
        "--reject-with",
        "tcp-reset",
    ]
    return spec


def rule_present(kind: str, spec: list[str]) -> bool:
    return run([firewall(), "-C", KINDS[kind], *spec], check=False).returncode == 0


def insert_rule(kind: str, spec: list[str]) -> bool:
    """Insert ``spec`` at the top of the kind's chain unless it is there."""

    if rule_present(kind, spec):
        return False
    run([firewall(), "-I", KINDS[kind], "1", *spec])
    return True


def tagged_rules(kind: str, rule_tag: str) -> list[str]:
    completed = run([firewall(), "-S", KINDS[kind]])
    needle = f"--comment {rule_tag} "
    return [
        line
        for line in completed.stdout.splitlines()
        if needle in line + " " and line.startswith(f"{APPEND} {KINDS[kind]} ")
    ]


def delete_tagged(kind: str, rule_tag: str) -> int:
    """Delete every rule of the chain carrying ``rule_tag``; the count."""

    deleted = 0
    for _ in range(MAX_DELETE_PASSES):
        rules = tagged_rules(kind, rule_tag)
        if not rules:
            return deleted
        for line in rules:
            parts = shlex.split(line)
            if parts[:2] != [APPEND, KINDS[kind]]:
                raise ProbeError(f"unexpected rule shape: {line[:120]}")
            run([firewall(), "-D", KINDS[kind], *parts[2:]])
            deleted += 1
    raise ProbeError("tagged rules keep reappearing")


def resolve_ipv4(host: str) -> list[str]:
    if SAFE_HOST.fullmatch(host) is None:
        raise ProbeError("unsafe endpoint host")
    try:
        return [str(ipaddress.IPv4Address(host))]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_INET)
    except socket.gaierror as exc:
        raise ProbeError(f"cannot resolve {host}: {exc}") from exc
    return sorted({str(item[4][0]) for item in infos})


def host_addresses() -> list[str]:
    """The node's own IPv4 addresses (``ip -4 -o addr``)."""

    completed = run(["ip", "-4", "-o", "addr", "show"], check=False)
    found = set()
    for line in completed.stdout.splitlines():
        match = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)/", line)
        if match is not None:
            found.add(match.group(1))
    return sorted(found)


def agent_listen_ports(unit: str) -> dict[str, Any]:
    """The Node Agent unit's state and the TCP ports its main process listens on."""

    state = unit_state(unit)
    main_pid = state.get("MainPID") or "0"
    ports: set[int] = set()
    addresses: set[str] = set()
    if main_pid not in {"", "0"}:
        completed = run(["ss", "-ltnpH"], check=False)
        for line in completed.stdout.splitlines():
            if f"pid={main_pid}," not in line:
                continue
            columns = line.split()
            if len(columns) < 4:
                continue
            local = columns[3]
            host_text, _, port_text = local.rpartition(":")
            if port_text.isdigit():
                ports.add(int(port_text))
            try:
                parsed = ipaddress.IPv4Address(host_text)
            except ValueError:
                continue
            if not parsed.is_unspecified and not parsed.is_loopback:
                addresses.add(str(parsed))
    return {
        "unit": unit,
        "active_state": state.get("ActiveState"),
        "main_pid": int(main_pid) if main_pid.isdigit() else None,
        "ports": sorted(ports),
        "addresses": sorted(addresses),
    }


def connect_target(agent: dict[str, Any] | None) -> str | None:
    """Where a local connectivity check reaches the Agent: the concrete IPv4 it
    listens on, else the node's first non-loopback address (the Agent binds the
    node's private IP by default, never loopback)."""

    for address in (agent or {}).get("addresses") or []:
        return str(address)
    for address in host_addresses():
        parsed = ipaddress.IPv4Address(address)
        if not parsed.is_loopback and not parsed.is_link_local:
            return address
    return None


def agent_connectivity(unit: str, port: int) -> bool | None:
    target = connect_target(agent_listen_ports(unit))
    return connectivity(target, port) if target else None


def connectivity(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / STATE_FILE
    if not path.is_file():
        return {"run_id": run_id, "blocks": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("blocks"), dict):
        raise ProbeError("isolation state is malformed")
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / STATE_FILE
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def survey(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    emit(
        {
            "run_id": run_id,
            "tools": {
                name: shutil.which(name) is not None
                for name in ("iptables", "systemd-run", "systemctl", "ss", "ip")
            },
            "agent": agent_listen_ports(arguments.unit),
            "host_addresses": host_addresses(),
            "existing_rules": {
                kind: tagged_rules(kind, tag(run_id, kind))
                for kind in KINDS
                if shutil.which("iptables")
            },
            "state": load_state(run_id),
        }
    )


def _pod_egress_targets(arguments: argparse.Namespace) -> dict[str, Any]:
    if not arguments.pod_ip or arguments.port is None:
        raise ProbeError("a pod-egress block needs --pod-ip and --port")
    try:
        pod_ip = str(ipaddress.IPv4Address(arguments.pod_ip))
    except ValueError as exc:
        raise ProbeError("the Pod IP is not an IPv4 address") from exc
    if pod_ip in host_addresses():
        raise ProbeError(
            "the Pod IP is one of the host's addresses (hostNetwork); refusing a "
            "node-wide block"
        )
    port = checked_port(arguments.port)
    cidrs = [
        str(ipaddress.IPv4Network(item, strict=False)) for item in arguments.cidr or []
    ]
    resolved = resolve_ipv4(arguments.host) if arguments.host else []
    destinations = cidrs or [f"{item}/32" for item in resolved]
    if not destinations:
        raise ProbeError("no destination to block: give --host or --cidr")
    return {
        "pod_ip": pod_ip,
        "host": arguments.host or None,
        "port": port,
        "resolved_ipv4": resolved,
        "destinations": destinations,
    }


def _agent_ingress_target(arguments: argparse.Namespace) -> dict[str, Any]:
    agent = agent_listen_ports(arguments.unit)
    if agent["active_state"] != "active":
        raise ProbeError(f"{arguments.unit} is not active ({agent['active_state']})")
    ports = list(agent["ports"])
    if arguments.port is not None:
        port = checked_port(arguments.port)
        if port not in ports:
            raise ProbeError(
                f"the Node Agent does not listen on port {port} (listening: {ports})"
            )
    elif len(ports) == 1:
        port = ports[0]
    else:
        raise ProbeError(f"cannot pick the Node Agent port from {ports}; give --port")
    return {"port": port, "unit": arguments.unit, "agent": agent}


def block(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    kind = checked_kind(arguments.kind)
    seconds = bounded(
        arguments.deadman_seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman"
    )
    rule_tag = tag(run_id, kind)
    firewall()
    if kind == "pod-egress":
        target = _pod_egress_targets(arguments)
        specs = [
            rule_spec(
                kind,
                rule_tag,
                port=target["port"],
                source=f"{target['pod_ip']}/32",
                destination=destination,
            )
            for destination in target["destinations"]
        ]
    else:
        target = _agent_ingress_target(arguments)
        specs = [rule_spec(kind, rule_tag, port=target["port"])]
    # Intent first, deadman second, rules last: a runner that dies between any
    # two of these leaves either nothing or a block that undoes itself.
    state = load_state(run_id)
    entry = dict(state["blocks"].get(kind) or {})
    entry.update(
        {
            "kind": kind,
            "chain": KINDS[kind],
            "tag": rule_tag,
            "target": {
                **(entry.get("target") or {}),
                **{key: value for key, value in target.items() if key != "agent"},
            },
            "deadman_seconds": seconds,
        }
    )
    state["blocks"][kind] = entry
    save_state(run_id, state)
    deadman = arm_deadman(run_id, kind, seconds)
    inserted = sum(insert_rule(kind, spec) for spec in specs)
    document: dict[str, Any] = {
        "run_id": run_id,
        "kind": kind,
        "chain": KINDS[kind],
        "tag": rule_tag,
        "target": target,
        "inserted": inserted,
        "rules": tagged_rules(kind, rule_tag),
        "deadman": deadman,
    }
    if kind == "agent-ingress":
        document["connectivity"] = agent_connectivity(arguments.unit, target["port"])
    emit(document)


def status(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    kind = checked_kind(arguments.kind)
    rule_tag = tag(run_id, kind)
    state = load_state(run_id)
    entry = state["blocks"].get(kind) or {}
    target = entry.get("target") or {}
    document: dict[str, Any] = {
        "run_id": run_id,
        "kind": kind,
        "tag": rule_tag,
        "rules": tagged_rules(kind, rule_tag),
        "timer": unit_state(f"{deadman_unit(run_id, kind)}.timer"),
        "recorded": entry,
    }
    port = target.get("port")
    if kind == "agent-ingress" and isinstance(port, int):
        document["connectivity"] = agent_connectivity(
            str(target.get("unit") or AGENT_UNIT), port
        )
    if kind == "pod-egress" and target.get("host"):
        resolved = resolve_ipv4(str(target["host"]))
        networks = [
            ipaddress.IPv4Network(str(item), strict=False)
            for item in target.get("destinations") or []
        ]
        document["resolved_ipv4"] = resolved
        # Addresses the endpoint resolves to now that no recorded destination
        # covers: a DNS change the runner answers with another ``block``.
        document["uncovered_ipv4"] = sorted(
            address
            for address in resolved
            if not any(ipaddress.IPv4Address(address) in net for net in networks)
        )
    emit(document)


def cleanup(arguments: argparse.Namespace) -> None:
    run_id = checked_run_id(arguments.run_id)
    kinds = list(KINDS) if arguments.kind == "all" else [checked_kind(arguments.kind)]
    state = load_state(run_id)
    report: dict[str, Any] = {}
    clean = True
    for kind in kinds:
        rule_tag = tag(run_id, kind)
        deleted = delete_tagged(kind, rule_tag) if shutil.which("iptables") else 0
        remaining = tagged_rules(kind, rule_tag) if shutil.which("iptables") else []
        timer = disarm_deadman(run_id, kind)
        entry = state["blocks"].pop(kind, None) or {}
        item: dict[str, Any] = {
            "deleted": deleted,
            "remaining": remaining,
            "timer": timer,
            "recorded": entry,
        }
        port = (entry.get("target") or {}).get("port")
        if kind == "agent-ingress" and isinstance(port, int):
            item["connectivity"] = agent_connectivity(
                str((entry.get("target") or {}).get("unit") or AGENT_UNIT), port
            )
        if remaining or timer.get("ActiveState") == "active":
            clean = False
        report[kind] = item
    save_state(run_id, state)
    if not state["blocks"]:
        for name in (STATE_FILE, "isolation-probe.py"):
            (state_dir(run_id) / name).unlink(missing_ok=True)
        try:
            state_dir(run_id).rmdir()
        except OSError:
            pass
    emit({"run_id": run_id, "kinds": report, "clean": clean})
    if arguments.strict and not clean:
        raise ProbeError("isolation cleanup left tagged rules or an armed timer")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = value.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Any) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.set_defaults(handler=handler)
        return item

    item = command("survey", survey)
    item.add_argument("--unit", default=AGENT_UNIT)
    item = command("block", block)
    item.add_argument("--kind", required=True, choices=sorted(KINDS))
    item.add_argument("--deadman-seconds", type=int, required=True)
    item.add_argument("--pod-ip", default="")
    item.add_argument("--host", default="")
    item.add_argument("--port", type=int, default=None)
    item.add_argument("--cidr", action="append", default=[])
    item.add_argument("--unit", default=AGENT_UNIT)
    item = command("status", status)
    item.add_argument("--kind", required=True, choices=sorted(KINDS))
    item = command("cleanup", cleanup)
    item.add_argument("--kind", default="all", choices=["all", *sorted(KINDS)])
    item.add_argument("--strict", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_code": hashlib.sha256(str(exc).encode()).hexdigest()[:16],
                "error_site": f"{frame.name}:{frame.lineno}",
                "command": arguments.command,
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
