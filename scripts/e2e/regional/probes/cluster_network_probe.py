#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import shlex
import subprocess
from typing import Any


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ProbeError(RuntimeError):
    pass


class MissingBinary(ProbeError):
    """A command the probe needs is not on this node (exit 127 territory).

    Live (mixed site, ISO-006 attempt 4): the block was aimed at the executor
    Pods' CPU nodes too, and the probe died "returned no JSON (exit 127)" with
    nothing naming what was missing. The binary's basename is a fixed token
    the fixture may show; the node's PATH and paths never leave the node.
    """

    def __init__(self, binary: str) -> None:
        super().__init__(f"required binary is missing on the node: {binary}")
        self.binary = binary


#: Every binary ``block`` touches, checked before the first mutation.
REQUIRED_BINARIES = ("iptables", "systemd-run", "systemctl")


def run(
    command: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except FileNotFoundError as exc:
        raise MissingBinary(command[0].rsplit("/", 1)[-1]) from exc
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): "
            f"{shlex.join(command)}: {completed.stderr.strip()}"
        )
    return completed


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def safe_id(value: str) -> str:
    if SAFE_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def normalized_cidrs(values: list[str]) -> list[str]:
    try:
        networks = {ipaddress.ip_network(value, strict=False) for value in values}
    except ValueError as exc:
        raise ProbeError("invalid control-plane CIDR") from exc
    if any(network.prefixlen == 0 for network in networks):
        raise ProbeError("default-route CIDR is prohibited")
    if not networks or any(network.version != 4 for network in networks):
        raise ProbeError("the IPv4 iptables probe requires explicit IPv4 CIDRs")
    return sorted(str(network) for network in networks)


def chain_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12].upper()
    return f"GFISO{digest}"


def restore_unit(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return f"gpu-fault-network-restore-{digest}"


# The executor and collector Pods are not hostNetwork: their egress leaves a
# veth and is routed by the host, so it traverses FORWARD, never OUTPUT. A
# jump from OUTPUT alone cut only the node's own processes (the Node Agent),
# and ISO-006 passed after thirty minutes in which cluster B was never
# actually offline. Both chains get the jump; both lose it on rollback.
HOST_CHAINS = ("OUTPUT", "FORWARD")


def remove_chain(chain: str, run_id: str, *, allow_empty: bool = False) -> None:
    state = chain_state(chain)
    if state["residual"] and (
        not state["rules"]
        and not allow_empty
        or any(
            "--comment" not in rule or rule[rule.index("--comment") + 1] != run_id
            for rule in state["rules"]
        )
    ):
        raise ProbeError("network chain ownership is unknown; refusing cleanup")
    for host_chain in HOST_CHAINS:
        run(["iptables", "-D", host_chain, "-j", chain], check=False)
    run(["iptables", "-F", chain], check=False)
    run(["iptables", "-X", chain], check=False)
    if chain_state(chain)["residual"]:
        raise ProbeError("network rules remain; restore timer must stay armed")


def chain_state(chain: str) -> dict[str, Any]:
    listing = run(["iptables", "-S"])
    rules = [shlex.split(line) for line in listing.stdout.splitlines() if line.strip()]
    present = any(rule[:2] == ["-N", chain] for rule in rules)
    jumps = {
        host: any(
            rule[:2] == ["-A", host] and rule[-2:] == ["-j", chain] for rule in rules
        )
        for host in HOST_CHAINS
    }
    own_rules = [rule for rule in rules if rule[:2] == ["-A", chain]]
    return {
        "blocked": present and bool(own_rules) and all(jumps.values()),
        "chain_present": present,
        "jumps": jumps,
        "rules": own_rules,
        "residual": present or bool(own_rules) or any(jumps.values()),
    }


def restore_script(chain: str) -> str:
    """The rollback the systemd timer runs: every jump, then the chain."""

    deletes = "; ".join(
        f"iptables -D {host_chain} -j {chain} || true" for host_chain in HOST_CHAINS
    )
    return f"{deletes}; iptables -F {chain} || true; iptables -X {chain} || true"


def require_binaries() -> None:
    """Refuse by name before the restore timer or any rule exists.

    Probed through ``run`` (``--version`` is read-only), so a binary that is
    not on the node surfaces as ``MissingBinary`` here rather than halfway
    through the block with a timer already armed.
    """

    for binary in REQUIRED_BINARIES:
        if binary == "iptables":
            run(["iptables", "-S"])
        else:
            run([binary, "--version"], check=False)


def block(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    cidrs = normalized_cidrs(arguments.control_plane_cidr)
    require_binaries()
    chain = chain_name(run_id)
    if chain_state(chain)["residual"]:
        raise ProbeError("network probe chain already exists; refusing to replace it")
    unit = restore_unit(run_id)
    script = restore_script(chain)
    run(
        [
            "systemd-run",
            "--unit",
            unit,
            f"--on-active={arguments.restore_seconds}s",
            "/bin/bash",
            "-ceu",
            script,
        ]
    )
    try:
        run(["iptables", "-N", chain])
        for cidr in cidrs:
            run(
                [
                    "iptables",
                    "-A",
                    chain,
                    "-p",
                    "tcp",
                    "-d",
                    cidr,
                    "--dport",
                    "443",
                    "-m",
                    "comment",
                    "--comment",
                    run_id,
                    "-j",
                    "REJECT",
                ]
            )
        for host_chain in HOST_CHAINS:
            run(["iptables", "-I", host_chain, "1", "-j", chain])
        state = chain_state(chain)
        if not state["blocked"]:
            raise ProbeError("network block could not be read back")
    except Exception:
        remove_chain(chain, run_id, allow_empty=True)
        run(["systemctl", "stop", unit + ".timer"], check=False)
        run(["systemctl", "reset-failed", unit + ".service"], check=False)
        raise
    emit(
        {
            "run_id": run_id,
            "chain": chain,
            "host_chains": list(HOST_CHAINS),
            "cidrs": cidrs,
            # The rules as the kernel read them back, so the evidence says what
            # was installed on this node rather than what was asked for.
            "rules": [shlex.join(rule) for rule in state["rules"]],
            "jumps": state["jumps"],
            "restore_unit": unit + ".timer",
            "restore_seconds": arguments.restore_seconds,
            "blocked": True,
        }
    )


def unblock(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    chain = chain_name(run_id)
    remove_chain(chain, run_id)
    unit = restore_unit(run_id)
    run(["systemctl", "stop", unit + ".timer"], check=False)
    run(["systemctl", "reset-failed", unit + ".service"], check=False)
    emit({"run_id": run_id, "chain": chain, **chain_state(chain)})


def status(arguments: argparse.Namespace) -> None:
    run_id = safe_id(arguments.run_id)
    chain = chain_name(run_id)
    emit(
        {
            "run_id": run_id,
            "chain": chain,
            **chain_state(chain),
        }
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    add = commands.add_parser("block")
    add.add_argument("--run-id", required=True)
    add.add_argument(
        "--control-plane-cidr",
        action="append",
        required=True,
        help="IPv4 CIDR the control-plane URL resolves to from this node (repeatable)",
    )
    add.add_argument("--restore-seconds", type=int, default=2100)
    add.set_defaults(handler=block)

    remove = commands.add_parser("unblock")
    remove.add_argument("--run-id", required=True)
    remove.set_defaults(handler=unblock)

    show = commands.add_parser("status")
    show.add_argument("--run-id", required=True)
    show.set_defaults(handler=status)
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
        seconds = getattr(arguments, "restore_seconds", 2100)
        if not 60 <= seconds <= 3600:
            raise ProbeError("restore seconds is outside 60..3600")
        arguments.handler(arguments)
    except Exception as exc:
        payload: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(exc, MissingBinary):
            payload["error_kind"] = "missing_binary"
            payload["missing_binary"] = exc.binary
        emit(payload)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
