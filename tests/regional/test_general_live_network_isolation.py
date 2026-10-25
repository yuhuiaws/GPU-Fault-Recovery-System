"""GF-GENERAL-LIVE network isolation: the probe on a fake host, the shared
mechanics and graders, and 028 ``api_unreachable`` over scripted records.

The probe's ``subprocess`` boundary is a recording fake host (iptables chains,
systemd timers, the Agent unit's ``ss`` listing, the node's addresses); the
tests pin the order intent -> deadman -> rules, the tag-scoped deletes, the
refusals before any host command, and the ``cleanup --strict`` contract the
deadman relies on. The 028 variant runs with a fake job, a fake watcher Pod
inventory and a phase-driven store so the whole sequence -- block, trigger,
emergency stop, frozen heartbeat, restore, single-action grading -- executes.
"""

from __future__ import annotations

import shlex
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_network_isolation as iso
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.probes import general_live_isolation_probe as probe
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._general_live_fakes import FakeProbe
from tests.regional._general_live_isolation_fakes import (
    NODE,
    PEER,
    POD_IP,
    T0,
    WATCHER_NODE,
    Regional,
    attempt_entry,
    attempt_records,
    heartbeat,
    isolation_context,
    kubectl_responder,
    probe_docs,
    watcher_items,
)

RUN_ID = "gl028-api-unreachable-a1-deadbeef"


# --------------------------------------------------------------------------- #
# the probe on a fake host
# --------------------------------------------------------------------------- #
def _completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


class FakeHost:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.chains: dict[str, list[str]] = {"INPUT": [], "FORWARD": []}
        self.timers: set[str] = set()
        self.started: dict[str, list[str]] = {}
        self.agent_active = "active"
        self.agent_pid = "4242"
        self.listen_ports = [9099]
        self.addresses = ["10.0.1.5"]
        self.sticky = False
        self.emitted: list[dict[str, Any]] = []

    def connectivity(self, _host: str, port: int) -> bool:
        return not any(
            f"--dport {port} " in f"{rule} " for rule in self.chains["INPUT"]
        )

    def run(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self.calls.append(command)
        name = Path(command[0]).name
        if name == "iptables":
            return self._iptables(command[1:])
        if name == "systemctl":
            return self._systemctl(command[1:])
        if name == "systemd-run":
            unit = next(p for p in command if p.startswith("--unit=")).split("=", 1)[1]
            self.timers.add(f"{unit}.timer")
            self.started[unit] = command
            return _completed()
        if name == "ss":
            lines = [
                f'LISTEN 0 128 {self.addresses[0]}:{port} 0.0.0.0:* users:(("python",pid={self.agent_pid},fd=7))'
                for port in self.listen_ports
            ]
            return _completed("\n".join(lines))
        if name == "ip":
            return _completed(
                "\n".join(
                    f"2: eth0    inet {address}/24 brd 10.0.1.255 scope global eth0"
                    for address in self.addresses
                )
            )
        return _completed()

    def _iptables(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        op, chain = args[0], args[1]
        rules = self.chains[chain]
        if op == "-S":
            lines = [f"-P {chain} ACCEPT"] + [f"-A {chain} {spec}" for spec in rules]
            return _completed("\n".join(lines))
        if op == "-C":
            return _completed() if shlex.join(args[2:]) in rules else _completed("", 1)
        if op == "-I":
            rules.insert(int(args[2]) - 1, shlex.join(args[3:]))
            return _completed()
        if op == "-D":
            spec = shlex.join(args[2:])
            if spec not in rules:
                return _completed("", 1)
            if not self.sticky:
                rules.remove(spec)
            return _completed()
        raise AssertionError(f"unexpected iptables call {args}")

    def _systemctl(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "show":
            unit = args[1]
            if unit.endswith(".timer"):
                active = unit in self.timers
                values = {
                    "ActiveState": "active" if active else "inactive",
                    "SubState": "waiting" if active else "dead",
                    "MainPID": "0",
                    "NextElapseUSecRealtime": "soon" if active else "",
                }
            elif probe.DEADMAN_PREFIX in unit:
                values = {"ActiveState": "inactive", "SubState": "dead", "MainPID": "0"}
            else:
                values = {
                    "ActiveState": self.agent_active,
                    "SubState": "running",
                    "MainPID": self.agent_pid if self.agent_active == "active" else "0",
                }
            return _completed("\n".join(f"{k}={v}" for k, v in values.items()))
        if args[0] == "stop":
            for unit in args[1:]:
                self.timers.discard(unit)
        return _completed()


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeHost:
    fake = FakeHost()
    monkeypatch.setattr(probe, "run", fake.run)
    monkeypatch.setattr(probe, "emit", fake.emitted.append)
    monkeypatch.setattr(probe, "connectivity", fake.connectivity)
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(probe.shutil, "which", lambda name: f"/usr/sbin/{name}")
    monkeypatch.setattr(
        probe,
        "resolve_ipv4",
        lambda host: ["203.0.113.10", "203.0.113.11"]
        if host == "cp.example.internal"
        else [host],
    )
    return fake


def _first(host: FakeHost, name: str) -> int:
    for index, call in enumerate(host.calls):
        if Path(call[0]).name == name:
            return index
    raise AssertionError(f"{name} was never called: {host.calls}")


def test_agent_ingress_block_records_arms_then_inserts_and_cleanup_proves_clean(
    host: FakeHost,
) -> None:
    argv = [
        "block",
        "--run-id",
        RUN_ID,
        "--kind",
        "agent-ingress",
        "--deadman-seconds",
        "900",
    ]
    assert probe.main(argv) == 0, host.emitted[-1]
    result = host.emitted[-1]
    tag = probe.tag(RUN_ID, "agent-ingress")
    assert result["target"]["port"] == 9099 and result["inserted"] == 1
    assert result["rules"] == [
        f"-A INPUT -p tcp --dport 9099 -m comment --comment {tag} -j REJECT --reject-with tcp-reset"
    ]
    assert result["connectivity"] is False, "the port rejects while blocked"
    assert result["deadman"]["timer"]["ActiveState"] == "active"
    deadman = host.started[probe.deadman_unit(RUN_ID, "agent-ingress")]
    assert "--on-active=900s" in deadman
    assert deadman[-7:] == [
        str(probe.probe_copy(RUN_ID)),
        "cleanup",
        "--run-id",
        RUN_ID,
        "--kind",
        "agent-ingress",
        "--strict",
    ], deadman
    state_file = probe.state_dir(RUN_ID) / probe.STATE_FILE
    assert state_file.is_file(), "the intent is recorded on the node"
    inserted = next(i for i, c in enumerate(host.calls) if c[1:3] == ["-I", "INPUT"])
    assert _first(host, "systemd-run") < inserted, "the deadman precedes the rule"

    assert probe.main(["status", "--run-id", RUN_ID, "--kind", "agent-ingress"]) == 0
    status = host.emitted[-1]
    assert len(status["rules"]) == 1 and status["connectivity"] is False
    assert status["timer"]["ActiveState"] == "active"

    assert probe.main(["cleanup", "--run-id", RUN_ID, "--kind", "agent-ingress"]) == 0
    cleaned = host.emitted[-1]
    assert cleaned["clean"] is True, cleaned
    item = cleaned["kinds"]["agent-ingress"]
    assert item["deleted"] == 1 and item["remaining"] == []
    assert item["connectivity"] is True and item["timer"]["ActiveState"] == "inactive"
    assert host.chains["INPUT"] == [] and not host.timers
    assert not state_file.exists(), "the record is gone with the last block"


def test_agent_ingress_refuses_an_inactive_unit_or_a_port_it_does_not_listen_on(
    host: FakeHost,
) -> None:
    host.agent_active = "inactive"
    argv = [
        "block",
        "--run-id",
        RUN_ID,
        "--kind",
        "agent-ingress",
        "--deadman-seconds",
        "900",
    ]
    assert probe.main(argv) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError"
    assert not any(c[1:2] == ["-I"] for c in host.calls), "no rule on a refusal"
    assert not host.timers, "no deadman on a refusal"

    host.agent_active = "active"
    assert probe.main([*argv, "--port", "9100"]) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError"
    assert host.chains["INPUT"] == []


def test_agent_ingress_needs_an_explicit_port_when_the_agent_listens_on_several(
    host: FakeHost,
) -> None:
    host.listen_ports = [9099, 9100]
    argv = [
        "block",
        "--run-id",
        RUN_ID,
        "--kind",
        "agent-ingress",
        "--deadman-seconds",
        "900",
    ]
    assert probe.main(argv) == 1
    assert probe.main([*argv, "--port", "9100"]) == 0
    assert "--dport 9100" in host.chains["INPUT"][0]


def test_pod_egress_block_scopes_by_pod_ip_and_covers_every_resolved_address(
    host: FakeHost,
) -> None:
    argv = [
        "block",
        "--run-id",
        RUN_ID,
        "--kind",
        "pod-egress",
        "--deadman-seconds",
        "540",
        "--pod-ip",
        POD_IP,
        "--host",
        "cp.example.internal",
        "--port",
        "443",
    ]
    assert probe.main(argv) == 0, host.emitted[-1]
    result = host.emitted[-1]
    tag = probe.tag(RUN_ID, "pod-egress")
    assert result["target"]["destinations"] == ["203.0.113.10/32", "203.0.113.11/32"]
    assert result["inserted"] == 2 and len(result["rules"]) == 2
    assert all(f"-s {POD_IP}/32 -d 203.0.113.1" in rule for rule in result["rules"]), (
        result["rules"]
    )
    assert all(
        f"--dport 443 -m comment --comment {tag}" in rule for rule in result["rules"]
    ), result["rules"]
    assert "connectivity" not in result, "no L4 probe from the Pod's namespace"

    # Idempotent: a second block (a re-resolution) adds nothing already there.
    assert probe.main(argv) == 0
    assert host.emitted[-1]["inserted"] == 0 and len(host.chains["FORWARD"]) == 2

    # A rule of another run shares the chain and survives this run's cleanup.
    host.chains["FORWARD"].append(
        "-s 10.0.40.78/32 -p tcp --dport 443 -m comment --comment gfrs-other-iso-pod-egress -j REJECT --reject-with tcp-reset"
    )
    assert probe.main(["cleanup", "--run-id", RUN_ID]) == 0
    cleaned = host.emitted[-1]
    assert cleaned["clean"] is True and cleaned["kinds"]["pod-egress"]["deleted"] == 2
    assert host.chains["FORWARD"] == [
        "-s 10.0.40.78/32 -p tcp --dport 443 -m comment --comment gfrs-other-iso-pod-egress -j REJECT --reject-with tcp-reset"
    ]


def test_pod_egress_status_names_addresses_the_endpoint_resolves_to_later(
    host: FakeHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [
        "block", "--run-id", RUN_ID, "--kind", "pod-egress", "--deadman-seconds", "540",
        "--pod-ip", POD_IP, "--cidr", "203.0.113.0/24", "--host", "cp.example.internal",
        "--port", "443",
    ]  # fmt: skip
    assert probe.main(argv) == 0, host.emitted[-1]
    assert host.emitted[-1]["target"]["destinations"] == ["203.0.113.0/24"]
    monkeypatch.setattr(
        probe, "resolve_ipv4", lambda host: ["203.0.113.10", "198.51.100.3"]
    )
    assert probe.main(["status", "--run-id", RUN_ID, "--kind", "pod-egress"]) == 0
    status = host.emitted[-1]
    assert status["uncovered_ipv4"] == ["198.51.100.3"], status
    assert status["recorded"]["target"]["pod_ip"] == POD_IP


@pytest.mark.parametrize(
    "argv",
    [
        [
            "block",
            "--run-id",
            RUN_ID,
            "--kind",
            "pod-egress",
            "--deadman-seconds",
            "540",
            "--pod-ip",
            "10.0.1.5",
            "--host",
            "cp.example.internal",
            "--port",
            "443",
        ],
        [
            "block",
            "--run-id",
            RUN_ID,
            "--kind",
            "pod-egress",
            "--deadman-seconds",
            "540",
            "--pod-ip",
            POD_IP,
            "--port",
            "443",
        ],
        [
            "block",
            "--run-id",
            RUN_ID,
            "--kind",
            "pod-egress",
            "--deadman-seconds",
            "540",
            "--host",
            "cp.example.internal",
            "--port",
            "443",
        ],
        [
            "block",
            "--run-id",
            RUN_ID,
            "--kind",
            "pod-egress",
            "--deadman-seconds",
            "540",
            "--pod-ip",
            "not-an-ip",
            "--host",
            "cp.example.internal",
            "--port",
            "443",
        ],
        [
            "block",
            "--run-id",
            RUN_ID,
            "--kind",
            "agent-ingress",
            "--deadman-seconds",
            "59",
        ],
        [
            "block",
            "--run-id",
            RUN_ID,
            "--kind",
            "agent-ingress",
            "--deadman-seconds",
            "5401",
        ],
        [
            "block",
            "--run-id",
            "Bad Run",
            "--kind",
            "agent-ingress",
            "--deadman-seconds",
            "900",
        ],
    ],  # fmt: skip
    ids=[
        "host-address-pod",
        "no-destination",
        "no-pod-ip",
        "bad-pod-ip",
        "deadman-short",
        "deadman-long",
        "run-id",
    ],
)
def test_block_refuses_unsafe_inputs_before_any_rule_or_timer(
    host: FakeHost, argv: list[str]
) -> None:
    assert probe.main(argv) == 1
    assert host.emitted[-1]["error_class"] == "ProbeError", host.emitted[-1]
    assert not any(c[1:2] in (["-I"], ["-D"]) for c in host.calls), host.calls
    assert not host.timers, host.timers


def test_strict_cleanup_fails_loudly_when_a_tagged_rule_cannot_be_deleted(
    host: FakeHost,
) -> None:
    argv = [
        "block",
        "--run-id",
        RUN_ID,
        "--kind",
        "agent-ingress",
        "--deadman-seconds",
        "900",
    ]
    assert probe.main(argv) == 0
    host.sticky = True
    assert (
        probe.main(
            ["cleanup", "--run-id", RUN_ID, "--kind", "agent-ingress", "--strict"]
        )
        == 1
    )
    error = host.emitted[-1]
    assert error["error_class"] == "ProbeError" and error["error_kind"] == "probe"
    assert len(error["error_code"]) == 16 and ":" in error["error_site"]
    # The non-strict cleanup the runner calls reports instead of raising.
    host.sticky = False
    host.chains["INPUT"].clear()
    assert probe.main(["cleanup", "--run-id", RUN_ID, "--kind", "agent-ingress"]) == 0
    assert host.emitted[-1]["clean"] is True


def test_survey_is_read_only_and_names_the_agent_port_and_existing_rules(
    host: FakeHost,
) -> None:
    assert probe.main(["survey", "--run-id", RUN_ID]) == 0
    survey = host.emitted[-1]
    assert survey["agent"] == {
        "unit": probe.AGENT_UNIT,
        "active_state": "active",
        "main_pid": 4242,
        "ports": [9099],
        "addresses": ["10.0.1.5"],
    }
    assert survey["host_addresses"] == ["10.0.1.5"]
    assert survey["existing_rules"] == {"agent-ingress": [], "pod-egress": []}
    assert not any(c[1:2] in (["-I"], ["-D"]) for c in host.calls), host.calls
    assert not any(Path(c[0]).name == "systemd-run" for c in host.calls), host.calls


def test_connectivity_targets_the_agents_listen_address_never_loopback(
    host: FakeHost,
) -> None:
    host.addresses = ["10.0.1.5", "127.0.0.1", "169.254.169.254"]
    agent = probe.agent_listen_ports(probe.AGENT_UNIT)
    assert agent["addresses"] == ["10.0.1.5"]
    assert probe.connect_target(agent) == "10.0.1.5"
    # A wildcard listener names no address: the node's first routable IPv4.
    host.addresses = ["127.0.0.1", "169.254.169.254", "10.0.1.9"]
    assert probe.connect_target({"addresses": []}) == "10.0.1.9"
    host.addresses = ["127.0.0.1"]
    assert probe.connect_target(None) is None


def test_rule_spec_and_tag_shapes() -> None:
    tag = probe.tag(RUN_ID, "agent-ingress")
    assert (
        tag.startswith("gfrs-")
        and tag.endswith("-iso-agent-ingress")
        and " " not in tag
    )
    spec = probe.rule_spec(
        "pod-egress", tag, port=443, source="10.0.0.1/32", destination="10.0.0.2/32"
    )
    assert spec[:4] == ["-s", "10.0.0.1/32", "-d", "10.0.0.2/32"]
    assert spec[-4:] == ["-j", "REJECT", "--reject-with", "tcp-reset"]
    with pytest.raises(probe.ProbeError):
        probe.rule_spec("agent-ingress", tag, port=70000)
    with pytest.raises(probe.ProbeError, match="unknown isolation kind"):
        probe.tag(RUN_ID, "egress")


# --------------------------------------------------------------------------- #
# the runner-side mechanics
# --------------------------------------------------------------------------- #


def test_watcher_pod_needs_exactly_one_ready_pod_with_its_own_ip() -> None:
    regional = Regional(kubectl_responder=kubectl_responder(watcher_items({})))
    pod = iso.watcher_pod(regional)
    assert pod == {
        "name": "gpu-fault-completion-watcher-abc",
        "uid": "uid-old",
        "resource_version": "100",
        "node": WATCHER_NODE,
        "pod_ip": POD_IP,
        "host_network": False,
        "phase": "Running",
        "ready": True,
    }
    assert regional.kubectl_calls[-1][1:] == (
        "get", "pod", "-l", f"app={iso.WATCHER_APP}", "-o", "json",
    )  # fmt: skip
    for inventory, message in [
        (watcher_items({}, {"uid": "uid-2"}), "exactly one Ready"),
        (watcher_items({"ready": False}), "exactly one Ready"),
        (watcher_items({"host_network": True}), "Pod IP of its own"),
        (watcher_items(), "exactly one Ready"),
    ]:
        with pytest.raises(VariantNotRun, match=message) as raised:
            iso.watcher_pod(Regional(kubectl_responder=kubectl_responder(inventory)))
        assert raised.value.reason == "FIXTURE_NOT_READY"


def test_control_plane_endpoint_reads_only_the_url_key_and_never_logs_it() -> None:
    regional = Regional(
        kubectl_responder=kubectl_responder(
            watcher_items({}), "https://cp.example.internal:8443/api/"
        )
    )
    assert iso.control_plane_endpoint(regional) == ("cp.example.internal", 8443)
    assert regional.kubectl_calls[-1][1:] == (
        "get", "secret", iso.CONNECTION_SECRET, "-o", "jsonpath={.data.control-plane-url}",
    )  # fmt: skip
    assert iso.endpoint_of("https://cp.example.internal") == (
        "cp.example.internal",
        443,
    )
    assert iso.endpoint_of("http://10.0.0.9/") == ("10.0.0.9", 80)
    with pytest.raises(RegionalFixtureError):
        iso.endpoint_of("ftp://cp")
    empty = Regional(kubectl_responder=lambda *_: "")
    with pytest.raises(RegionalFixtureError, match="carries no control-plane-url"):
        iso.control_plane_endpoint(empty)


def test_agent_port_and_block_budget() -> None:
    assert iso.agent_port({"endpoint": "https://10.0.43.6:9099"}) == 9099
    assert iso.agent_port({"endpoint": "https://10.0.43.6"}) == 443
    with pytest.raises(VariantNotRun):
        iso.agent_port(None)
    with pytest.raises(VariantNotRun):
        iso.agent_port({"endpoint": ""})


def test_block_budget_stays_inside_the_window_and_the_ceiling(tmp_path: Path) -> None:
    regional = Regional(kubectl_responder=kubectl_responder(watcher_items({})))
    ctx = isolation_context(tmp_path, regional, probes={}, window_minutes=120)
    assert iso.block_budget(ctx, iso.WATCHER_BLOCK_SECONDS) == iso.WATCHER_BLOCK_SECONDS
    short = isolation_context(tmp_path / "short", regional, probes={}, window_minutes=5)
    assert (
        iso.block_budget(short, iso.AGENT_BLOCK_SECONDS) == 300 - iso.MIN_BLOCK_SECONDS
    )
    tiny = isolation_context(tmp_path / "tiny", regional, probes={}, window_minutes=1)
    assert iso.block_budget(tiny, iso.AGENT_BLOCK_SECONDS) == iso.MIN_BLOCK_SECONDS


def test_host_isolation_registers_its_undo_first_and_passes_the_target(
    tmp_path: Path,
) -> None:
    regional = Regional(kubectl_responder=kubectl_responder(watcher_items({})))
    fake = FakeProbe(probe_docs("pod-egress"))
    ctx = isolation_context(
        tmp_path, regional, probes={f"isolation@{WATCHER_NODE}": fake}
    )
    block = iso.HostNetworkIsolation(
        ctx,
        node=WATCHER_NODE,
        kind=iso.POD_EGRESS,
        pod_ip=POD_IP,
        host="cp.example.internal",
        port=443,
        cidrs=("203.0.113.0/24",),
        deadman_seconds=540,
    )
    block.block()
    assert ctx.cleanup.pending() == [block.cleanup_name]
    assert fake.calls[-1] == (
        "block", "--kind", "pod-egress", "--deadman-seconds", "540", "--pod-ip", POD_IP,
        "--port", "443", "--host", "cp.example.internal", "--cidr", "203.0.113.0/24",
        "--run-id", ctx.run_id,
    )  # fmt: skip
    assert block.blocked_at is not None
    assert (ctx.evidence.case_dir / "isolation-pod-egress-block.json").is_file(), (
        "the block document is evidence"
    )
    block.status()
    block.restore()
    assert fake.calls[-1] == ("cleanup", "--kind", "pod-egress", "--run-id", ctx.run_id)
    assert block.restored_at is not None and block.restored is not None

    agent = iso.HostNetworkIsolation(
        ctx, node=NODE, kind=iso.AGENT_INGRESS, port=9099, deadman_seconds=2700
    )
    ctx.probes[f"isolation@{NODE}"] = FakeProbe(probe_docs("agent-ingress"))  # type: ignore[assignment]
    agent.block()
    assert ctx.probes[f"isolation@{NODE}"].calls[-1][:7] == (  # type: ignore[attr-defined]
        "block", "--kind", "agent-ingress", "--deadman-seconds", "2700", "--unit", iso.AGENT_UNIT,
    )  # fmt: skip
    with pytest.raises(ValueError, match="pod-egress block needs"):
        iso.HostNetworkIsolation(
            ctx, node=NODE, kind=iso.POD_EGRESS, deadman_seconds=300
        )
    with pytest.raises(ValueError, match="unknown isolation kind"):
        iso.HostNetworkIsolation(ctx, node=NODE, kind="egress", deadman_seconds=300)


def test_host_isolation_refuses_a_half_block_and_an_unproven_cleanup(
    tmp_path: Path,
) -> None:
    regional = Regional(kubectl_responder=kubectl_responder(watcher_items({})))
    docs = probe_docs("agent-ingress")
    docs["block"] = {
        "rules": [],
        "deadman": {"timer": {"ActiveState": "inactive"}},
        "connectivity": True,
    }
    docs["cleanup"] = {
        "clean": False,
        "kinds": {
            "agent-ingress": {
                "remaining": ["-A INPUT x"],
                "timer": {"ActiveState": "active"},
                "connectivity": False,
            }
        },
    }
    ctx = isolation_context(
        tmp_path, regional, probes={f"isolation@{NODE}": FakeProbe(docs)}
    )
    block = iso.HostNetworkIsolation(
        ctx, node=NODE, kind=iso.AGENT_INGRESS, port=9099, deadman_seconds=300
    )
    with pytest.raises(RegionalFixtureError, match="not established") as raised:
        block.block()
    message = str(raised.value)
    assert "no tagged rule" in message and "deadman timer is not active" in message
    assert "still accepts connections" in message
    assert ctx.cleanup.pending() == [block.cleanup_name], (
        "the undo is registered anyway"
    )
    with pytest.raises(
        RegionalFixtureError, match="did not prove the node clean"
    ) as raised:
        block.restore()
    message = str(raised.value)
    assert "tagged rules remain" in message and "still armed" in message
    assert "does not accept connections" in message


def test_block_and_cleanup_error_shapes() -> None:
    assert (
        iso.block_errors(
            {"rules": ["r"], "deadman": {"timer": {"ActiveState": "active"}}},
            iso.POD_EGRESS,
        )
        == []
    )
    assert (
        iso.cleanup_errors(
            {"clean": True, "kinds": {"pod-egress": {"remaining": []}}}, iso.POD_EGRESS
        )
        == []
    )
    assert iso.cleanup_errors({"clean": True, "kinds": {}}, iso.AGENT_INGRESS) == [
        "the Node Agent port does not accept connections after the restore (connectivity=None)"
    ]


# --------------------------------------------------------------------------- #
# graders (pure)
# --------------------------------------------------------------------------- #


def test_ineffective_block_reasons_read_the_heartbeat_and_the_terminal() -> None:
    blocked_at = T0 + timedelta(minutes=5)
    quiet = {"heartbeat": heartbeat(T0)}
    assert iso.ineffective_block_reasons(quiet, {}, blocked_at=blocked_at) == []
    assert (
        iso.ineffective_block_reasons({"heartbeat": None}, {}, blocked_at=blocked_at)
        == []
    )
    late = {"heartbeat": heartbeat(blocked_at + timedelta(seconds=30))}
    reasons = iso.ineffective_block_reasons(
        late, {"event": {"x": 1}}, blocked_at=blocked_at
    )
    assert (
        len(reasons) == 2
        and "still reached" in reasons[0]
        and "terminal reached" in reasons[1]
    )


def test_emergency_stop_errors_need_the_mode_and_the_containment_incident() -> None:
    workload = {
        "metadata": {
            "annotations": {
                iso.EMERGENCY_MODE_ANNOTATION: iso.EMERGENCY_MODE,
                iso.TERMINATION_ANNOTATION: "inc-c",
            }
        }
    }
    assert iso.emergency_stop_errors(workload, incident_id="inc-c") == []
    assert iso.emergency_stop_errors(workload) == []
    assert iso.emergency_stop_errors(workload, incident_id="inc-other") == [
        "the workload's termination initiator inc-c is not the attempt's containment incident inc-other"
    ]
    errors = iso.emergency_stop_errors({"metadata": {"annotations": {}}})
    assert (
        len(errors) == 2
        and "no emergency-fallback" in errors[0]
        and "no termination-initiator" in errors[1]
    )


def test_single_action_errors_count_every_action_once() -> None:
    entry = attempt_entry()
    assert iso.single_action_errors(entry, attempt_records(entry)) == []
    assert iso.single_action_errors(attempt_entry(stops=2), attempt_records(entry)) == [
        "STOP_WORKLOADS succeeded 2 times for one attempt"
    ]
    assert iso.single_action_errors(attempt_entry(stops=0), attempt_records(entry)) == [
        "the containment workflow's STOP_WORKLOADS never succeeded"
    ]
    assert iso.single_action_errors(
        attempt_entry(recovery=2), attempt_records(entry)
    ) == [
        "2 recovery workflows for one attempt",
        "RESTART_WORKLOAD succeeded 2 times for one attempt",
    ]
    assert iso.single_action_errors(
        attempt_entry(restarts=2), attempt_records(entry)
    ) == ["RESTART_WORKLOAD succeeded 2 times for one attempt"]
    assert iso.single_action_errors(
        attempt_entry(decision=None), attempt_records(entry)
    ) == ["no completion decision was recorded after the restore"]
    doubled = attempt_records(entry, count=2, reservations=["a", "b"])
    assert iso.single_action_errors(entry, doubled) == [
        "restart_count 2 exceeds the one budgeted restart"
    ]
    skewed = attempt_records(entry, count=1, reservations=[])
    assert iso.single_action_errors(entry, skewed) == [
        "restart_count does not match the reservation ids"
    ]


def test_heartbeat_notification_incident_and_observation_helpers() -> None:
    restored_at = T0 + timedelta(minutes=10)
    assert (
        iso.restored_heartbeat_errors(
            {"heartbeat": heartbeat(restored_at + timedelta(seconds=90))},
            restored_at=restored_at,
        )
        == []
    )
    assert iso.restored_heartbeat_errors(
        {"heartbeat": None}, restored_at=restored_at
    ) == ["no watcher coverage heartbeat is recorded after the restore"]
    stale = iso.restored_heartbeat_errors(
        {"heartbeat": heartbeat(T0)}, restored_at=restored_at
    )
    assert stale[0].startswith("the watcher's last coverage heartbeat"), stale
    notifications = [
        {"incident_id": "inc-c", "category": "containment", "deduplication_key": "k1"},
        {"incident_id": "inc-c", "category": "containment", "deduplication_key": "k1"},
        {"incident_id": "inc-r", "category": "restart", "deduplication_key": "k2"},
        {"incident_id": "inc-r", "category": "restart", "deduplication_key": "k3"},
    ]
    errors = iso.duplicate_notification_errors(notifications)
    assert errors == [
        "notification k1 was recorded 2 times",
        "2 containment notifications for incident inc-c",
        "2 restart notifications for incident inc-r",
    ]
    assert iso.duplicate_notification_errors(notifications[2:3]) == []
    entry = attempt_entry(recovery=2)
    assert iso.attempt_incident_ids(entry) == ["inc-c", "inc-r0", "inc-r1"]
    records = {
        "observations": [
            {"attempt_id": "job-a001", "workload_phase": "PENDING"},
            {"attempt_id": "job-a001", "workload_phase": "RUNNING", "containers": 1},
        ]
    }
    assert iso.observed_running(records, "job-a001") == {
        "attempt_id": "job-a001",
        "workload_phase": "RUNNING",
        "containers": 1,
    }
    assert iso.observed_running(records, "job-a002") is None


def _hung_workflow(
    *,
    summary: dict[str, Any],
    triage: dict[str, Any],
    later: tuple[tuple[str, str], ...] = (
        ("COLLECT_DIAGNOSTIC_BUNDLE", "SUCCEEDED"),
        ("VALIDATE_FABRIC", "SUCCEEDED"),
        ("ESCALATE_SUPPORT", "SUCCEEDED"),
    ),
    status: str = "SUCCEEDED",
    acted: tuple[str, ...] = (),
) -> dict[str, Any]:
    ops = [
        "FREEZE_EVIDENCE",
        "COLLECT_HUNG_SUMMARY",
        "COLLECT_HUNG_SUMMARY",
        "COLLECT_HUNG_TRIAGE",
        "COLLECT_HUNG_TRIAGE",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "VALIDATE_FABRIC",
        "ESCALATE_SUPPORT",
    ]
    executions = [
        {
            "step_index": 0,
            "operation": "FREEZE_EVIDENCE",
            "status": "SUCCEEDED",
            "details": {},
        },
        {
            "step_index": 1,
            "operation": "COLLECT_HUNG_SUMMARY",
            "status": "SUCCEEDED",
            "details": summary,
        },
        {
            "step_index": 3,
            "operation": "COLLECT_HUNG_TRIAGE",
            "status": "SUCCEEDED",
            "details": triage,
        },
        {
            "step_index": 4,
            "operation": "COLLECT_HUNG_TRIAGE",
            "status": "SUCCEEDED",
            "details": {"node_results": {}},
        },
    ]
    executions += [
        {"step_index": ops.index(op), "operation": op, "status": st, "details": {}}
        for op, st in later
    ]
    executions += [
        {"step_index": 9, "operation": op, "status": "SUCCEEDED", "details": {}}
        for op in acted
    ]
    return {
        "request_id": "wf-hung",
        "status": status,
        "official_steps": [{"operation": op, "parameters": {}} for op in ops],
        "step_executions": executions,
    }


def test_coverage_gap_errors_need_the_gap_recorded_without_a_fabricated_reading() -> (
    None
):
    good = _hung_workflow(
        summary={
            "node_results": {NODE: "enc"},
            "undetermined_nodes": [PEER],
            "node_failures": {PEER: "NODE_TIMEOUT"},
        },
        triage={"node_results": {NODE: "enc"}, "undetermined_nodes": [PEER]},
    )
    assert iso.coverage_gap_errors(good, node=PEER, peer=NODE) == []
    via_decision = _hung_workflow(
        summary={"node_results": {NODE: "enc"}},
        triage={
            "node_results": {NODE: "enc"},
            "hung_triage_decision": {"not_sampled_nodes": [PEER]},
        },
    )
    assert iso.coverage_gap_errors(via_decision, node=PEER, peer=NODE) == []
    via_coverage = _hung_workflow(
        summary={
            "node_results": {NODE: "enc"},
            "hung_survey": {
                "survey_coverage": {
                    "expected_node_count": 2,
                    "responding_node_count": 1,
                }
            },
        },
        triage={"node_results": {NODE: "enc"}},
    )
    assert iso.coverage_gap_errors(via_coverage, node=PEER, peer=NODE) == []
    silent = _hung_workflow(
        summary={"node_results": {NODE: "enc"}}, triage={"node_results": {NODE: "enc"}}
    )
    assert iso.coverage_gap_errors(silent, node=PEER, peer=NODE) == [
        f"no summary/triage record names {PEER} as undetermined, failed or not responding (the coverage gap is not recorded)"
    ]
    fabricated = _hung_workflow(
        summary={
            "node_results": {NODE: "enc", PEER: "zero"},
            "undetermined_nodes": [PEER],
        },
        triage={"node_results": {NODE: "enc"}},
    )
    errors = iso.coverage_gap_errors(fabricated, node=PEER, peer=NODE)
    assert errors == [
        f"{PEER} has a result in ['COLLECT_HUNG_SUMMARY'] although its Agent was unreachable (a fabricated reading)"
    ]
    no_peer = _hung_workflow(
        summary={"node_results": {}, "undetermined_nodes": [PEER]},
        triage={"node_results": {}},
    )
    assert iso.coverage_gap_errors(no_peer, node=PEER, peer=NODE) == [
        f"no summary/triage evidence was collected from the reachable {NODE}"
    ]
    assert iso.coverage_gap_errors({"step_executions": []}, node=PEER, peer=NODE) == [
        "no COLLECT_HUNG_SUMMARY/COLLECT_HUNG_TRIAGE execution was recorded"
    ]


def test_evidence_recorded_and_later_step_statuses() -> None:
    assert iso.evidence_recorded({}) is False
    done = _hung_workflow(summary={}, triage={})
    assert iso.evidence_recorded(done) is True
    assert iso.later_step_statuses(done) == {
        "COLLECT_DIAGNOSTIC_BUNDLE": "SUCCEEDED",
        "VALIDATE_FABRIC": "SUCCEEDED",
        "ESCALATE_SUPPORT": "SUCCEEDED",
    }
    pending = _hung_workflow(summary={}, triage={}, later=(), status="RUNNING")
    assert iso.evidence_recorded(pending) is True, "both triage waves have ended"
    pending["step_executions"] = pending["step_executions"][:3]
    assert iso.evidence_recorded(pending) is False, "wave 2 has not ended"
    assert iso.later_step_statuses(pending) == {
        "COLLECT_DIAGNOSTIC_BUNDLE": None,
        "VALIDATE_FABRIC": None,
        "ESCALATE_SUPPORT": None,
    }
    pending["status"] = "FAILED"
    assert iso.evidence_recorded(pending) is True, "a terminal workflow is final"
    only_later = {
        "status": "RUNNING",
        "official_steps": [{"operation": "VALIDATE_FABRIC"}],
        "step_executions": [
            {"step_index": 0, "operation": "VALIDATE_FABRIC", "status": "WAITING"}
        ],
    }
    assert iso.evidence_recorded(only_later) is True


def test_no_isolation_and_channel_restored_errors() -> None:
    clean = _hung_workflow(summary={}, triage={})
    node = {
        "name": PEER,
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
    }
    assert iso.no_isolation_errors(clean, node) == []
    acted = _hung_workflow(summary={}, triage={}, acted=("STOP_WORKLOADS", "RESET_GPU"))
    errors = iso.no_isolation_errors(acted, {**node, "unschedulable": True})
    assert (
        errors[0] == "the hung chain executed forbidden ['RESET_GPU', 'STOP_WORKLOADS']"
    )
    assert errors[1] == f"{PEER} was isolated on a hung suspicion"
    restored_at = T0 + timedelta(minutes=30)
    cleanup = {
        "clean": True,
        "kinds": {
            "agent-ingress": {
                "remaining": [],
                "timer": {"ActiveState": "inactive"},
                "connectivity": True,
            }
        },
    }
    before = {"last_seen_at": T0.isoformat(), "lifecycle_state": "ACTIVE"}
    after = {
        "last_seen_at": (restored_at + timedelta(seconds=40)).isoformat(),
        "lifecycle_state": "ACTIVE",
    }
    assert (
        iso.channel_restored_errors(cleanup, before, after, restored_at=restored_at)
        == []
    )
    assert iso.channel_restored_errors(
        cleanup, before, None, restored_at=restored_at
    ) == ["the node's Agent record is gone after the restore"]
    stale = {"last_seen_at": T0.isoformat(), "lifecycle_state": "DRAINING"}
    errors = iso.channel_restored_errors(
        cleanup, before, stale, restored_at=restored_at
    )
    assert errors == [
        "the Agent is DRAINING after the restore",
        f"the Agent's heartbeat did not advance after the restore (last_seen_at={T0.isoformat()})",
        "the Agent's last_seen_at did not move during the variant",
    ]
