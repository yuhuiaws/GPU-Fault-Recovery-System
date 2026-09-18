from __future__ import annotations

import io
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from threading import Event
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied, process_identity
from scripts.e2e.regional.probes import late_ownership_node_probe as probe
from tests.regional._late_ownership_support import evidence, scope


@pytest.mark.parametrize(
    "cgroup,matched",
    [
        ("0::/kubepods/burstable/podabc-def/container\n", True),
        ("0::/kubepods.slice/kubepods-burstable-podabc_def.slice/container\n", True),
        ("0::/kubepods/podabc-def-not-this-one/container\n", False),
        ("0::/system.slice/gpu-fault-node-agent.service\n", False),
    ],
)
def test_exact_pod_cgroup_identity(cgroup, matched):
    assert probe.matches_pod(cgroup, "abc-def") is matched


def test_malformed_cgroup_is_not_physical_absence():
    with pytest.raises(BoundaryDenied, match="malformed"):
        probe.matches_pod("unknown", "abc")


def test_compute_client_receipt_binds_pid_incarnation_and_exact_pod(
    tmp_path, monkeypatch
):
    binding = scope()
    pid = 321
    directory = tmp_path / str(pid)
    directory.mkdir()
    (directory / "cgroup").write_text("0::/kubepods/podowned-pod/container\n")
    identity = process_identity(os.getpid())
    monkeypatch.setattr(probe, "process_identity", lambda *a, **kw: identity)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=f"GPU-a, {pid}, python\n\n")

    observed = probe.observe_clients(
        binding,
        binding.nodes[0],
        ("owned-pod", "other-pod"),
        runner=runner,
        proc=tmp_path,
    )
    assert calls[0][0] == list(probe.CALIBRATION)
    assert calls[0][1]["check"] is True and calls[0][1]["timeout"] == 15
    assert observed["observations"][0]["pod_uids"] == ["owned-pod"]
    assert observed["observations"][0]["process"] == identity.model_dump(mode="json")
    assert len(observed["observations"][0]["cgroup_sha256"]) == 64
    assert "python" not in json.dumps(observed), (
        "raw process names must not leave the witness"
    )
    absent = probe.observe_clients(
        binding, binding.nodes[0], ("not-this-pod",), runner=runner, proc=tmp_path
    )
    assert absent["observations"] == []


@pytest.mark.parametrize(
    "csv", ["wrong,row\n", ",123,python\n", "GPU-a,notpid,python\n"]
)
def test_bad_nvidia_output_cannot_report_zero_clients(csv):
    binding = scope()
    with pytest.raises(BoundaryDenied, match="malformed"):
        probe.observe_clients(
            binding,
            binding.nodes[0],
            ("pod",),
            runner=lambda *a, **kw: SimpleNamespace(stdout=csv),
        )


def test_reused_gpu_client_pid_cannot_pass_observation(tmp_path, monkeypatch):
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "cgroup").write_text("0::/podowned\n")
    current = process_identity(os.getpid())
    identities = iter(
        [current, current.model_copy(update={"start_ticks": current.start_ticks + 1})]
    )
    monkeypatch.setattr(probe, "process_identity", lambda *a, **kw: next(identities))
    binding = scope()
    with pytest.raises(BoundaryDenied, match="changed"):
        probe.observe_clients(
            binding,
            binding.nodes[0],
            ("owned",),
            proc=tmp_path,
            runner=lambda *a, **kw: SimpleNamespace(stdout="GPU-a,1,python\n"),
        )


@dataclass
class WireReceipt:
    value: dict
    quiescence_sha256: str | None = None

    def model_dump(self, **kwargs):
        return self.value


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    proof = evidence()
    owner = process_identity(os.getpid())
    node = proof.scope.nodes[0].model_copy(update={"boot_id": owner.boot_id})
    binding = proof.scope.model_copy(update={"nodes": (node, proof.scope.nodes[1])})
    directory = tmp_path / "mailboxes" / "node-a"
    monkeypatch.setattr(probe, "mailbox_path", lambda *a: directory)
    monkeypatch.setenv("LATE_OWNERSHIP_WITNESS_POD_UID", "owned-witness-pod")
    monkeypatch.setattr(probe.shutil, "which", lambda name: "/usr/bin/true")

    def read_pid(command, **kwargs):
        assert command == [
            "systemctl",
            "show",
            "--property=MainPID",
            "--value",
            "gpu-fault-node-agent.service",
        ]
        return SimpleNamespace(stdout=str(os.getpid()))

    monkeypatch.setattr(probe.subprocess, "run", read_pid)
    ready, calls = Event(), []

    class Witness:
        def __init__(self, path, tracee, **kwargs):
            assert path == directory and tracee == owner

        def start(self):
            calls.append("attach")

        def check(self):
            ready.set()
            calls.append("continuous")

        def start_receipt(self, *args, **kwargs):
            calls.append("calibrate")
            return WireReceipt({"calibrated": True})

        def finish_receipt(self, start, quiet, **kwargs):
            calls.append("finish")
            return WireReceipt({"complete": True}, quiet.digest())

        def close(self):
            calls.append("close")

    monkeypatch.setattr(probe, "AttachedExecWitness", Witness)
    monkeypatch.setattr(
        probe, "observe_clients", lambda *a: {"observations": [{"pid": "physical"}]}
    )
    pool = ThreadPoolExecutor(max_workers=1)
    child = probe.NodeProbe(binding, node)
    future = pool.submit(child.run_daemon)
    assert ready.wait(2), "owned node witness never reached socket readiness"

    def request(action, payload=None):
        return probe.daemon_request(binding, node, action, payload or {})

    try:
        yield SimpleNamespace(
            request=request,
            binding=binding,
            node=node,
            future=future,
            directory=directory,
            owner=owner,
            calls=calls,
            quiet=proof.quiescence,
        )
    finally:
        if not future.done():
            try:
                request("invalid-terminal-command")
            except (OSError, ValueError, BoundaryDenied):
                pass
        try:
            future.result(2)
        except BoundaryDenied:
            pass
        pool.shutdown(wait=True)
        assert future.done(), "owned daemon thread was not drained"


def test_owned_daemon_stays_independent_between_requests_and_drains(daemon):
    status = daemon.request("status")
    assert status["phase"] == "ATTACHED"
    assert status["peer"] == daemon.owner.model_dump(mode="json")
    assert daemon.request("calibrated")["receipt"] == {"calibrated": True}
    assert daemon.request("status")["phase"] == "ARMED"
    assert not daemon.future.done(), (
        "RPC completion must not end the continuous witness"
    )
    clients = daemon.request("clients", {"pod_uids": ["owned"]})
    assert clients["observation"]["observations"] == [{"pid": "physical"}]
    quiet = daemon.quiet.model_dump(mode="json")
    assert daemon.request("finish", quiet)["receipt"] == {"complete": True}
    assert daemon.request("finish", quiet)["receipt"] == {"complete": True}
    assert daemon.calls.count("finish") == 1
    assert daemon.request("status")["phase"] == "COMPLETE"
    assert daemon.request("exit")["exited"] is True
    daemon.future.result(2)
    assert daemon.calls[-1] == "close"
    assert not daemon.directory.exists(), (
        "drained daemon cleanup must remove its owned mailbox directory"
    )


@pytest.mark.parametrize(
    "action,payload",
    [
        ("exit", {}),
        ("clients", {"pod_uids": ["owned"]}),
        ("finish", {}),
        ("unsupported", {}),
    ],
)
def test_out_of_order_mailbox_commands_fail_closed(daemon, action, payload):
    with pytest.raises((ValueError, BoundaryDenied)):
        daemon.request(action, payload)
    with pytest.raises(BoundaryDenied):
        daemon.future.result(2)
    assert "calibrate" not in daemon.calls and daemon.calls[-1] == "close"


@pytest.mark.parametrize("uids", [None, [], [""], [12], "pod"])
def test_client_observation_requires_explicit_nonempty_uid_set(daemon, uids):
    daemon.request("calibrated")
    with pytest.raises((ValueError, BoundaryDenied)):
        daemon.request("clients", {"pod_uids": uids})
    with pytest.raises(BoundaryDenied, match="UID"):
        daemon.future.result(2)


def test_finish_for_a_different_quiescence_cannot_reuse_a_trace(daemon):
    daemon.request("calibrated")
    daemon.request("finish", daemon.quiet.model_dump(mode="json"))
    other = daemon.quiet.model_copy(update={"sequence": daemon.quiet.sequence + 1})
    with pytest.raises((ValueError, BoundaryDenied)):
        daemon.request("finish", other.model_dump(mode="json"))
    with pytest.raises(BoundaryDenied, match="another"):
        daemon.future.result(2)


def test_stale_scope_request_cannot_command_the_live_witness(daemon):
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
        client.settimeout(2)
        client.connect(str(daemon.directory / "control.sock"))
        client.sendall(
            json.dumps(
                {
                    "scope_sha256": "0" * 64,
                    "request_id": "1" * 32,
                    "action": "calibrated",
                    "payload": {},
                }
            ).encode()
        )
        assert client.recv(1024) == b""
    with pytest.raises(BoundaryDenied, match="unbound"):
        daemon.future.result(2)
    assert "calibrate" not in daemon.calls


def test_daemon_has_no_host_mutation_without_downward_api_identity(
    tmp_path, monkeypatch
):
    binding = scope()
    monkeypatch.delenv("LATE_OWNERSHIP_WITNESS_POD_UID", raising=False)
    monkeypatch.setattr(probe, "mailbox_path", lambda *a: tmp_path / "not-created")
    with pytest.raises(BoundaryDenied, match="Pod UID"):
        probe.NodeProbe(binding, binding.nodes[0]).run_daemon()
    assert not list(tmp_path.iterdir()), (
        "missing Pod UID must not create host-side artifacts"
    )


@pytest.fixture
def stale_mailbox(tmp_path, monkeypatch):
    binding = scope()
    directory = tmp_path / "mailboxes" / "node-a"
    directory.mkdir(mode=0o700, parents=True)
    directory.parent.chmod(0o700)
    monkeypatch.setattr(probe, "mailbox_path", lambda *a: directory)
    owner = {
        "scope_sha256": binding.digest(),
        "pod_uid": "owned",
        "producer": {
            "pid": 999999999,
            "uid": os.geteuid(),
            "start_ticks": 1,
            "boot_id": "test",
        },
    }
    receipt = directory / "owner.json"
    receipt.write_text(json.dumps(owner))
    receipt.chmod(0o600)
    return binding, directory, owner


def test_owned_dead_daemon_mailbox_can_be_removed_without_signalling(stale_mailbox):
    binding, directory, _owner = stale_mailbox
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
        server.bind(str(directory / "control.sock"))
    probe.cleanup_mailbox(binding, binding.nodes[0], "owned")
    assert not directory.exists(), "cleanup must remove the dead daemon's owned mailbox"
    probe.cleanup_mailbox(binding, binding.nodes[0], "owned")


@pytest.mark.parametrize(
    "defect", ["uid", "scope", "public", "symlink", "foreign", "socket", "alive"]
)
def test_mailbox_cleanup_refuses_wrong_uid_replacement_or_live_process(
    stale_mailbox, defect
):
    binding, directory, owner = stale_mailbox
    if defect == "uid":
        owner["pod_uid"] = "replaced"
    elif defect == "scope":
        owner["scope_sha256"] = "f" * 64
    elif defect == "public":
        directory.chmod(0o777)
    elif defect == "foreign":
        (directory / "foreign").write_text("owned by another actor")
    elif defect == "socket":
        (directory / "control.sock").write_text("replaced")
    elif defect == "alive":
        owner["producer"] = process_identity(os.getpid()).model_dump(mode="json")
    (directory / "owner.json").write_text(json.dumps(owner))
    if defect == "symlink":
        (directory / "owner.json").rename(directory / "target")
        (directory / "owner.json").symlink_to(directory / "target")
    with pytest.raises(BoundaryDenied):
        probe.cleanup_mailbox(binding, binding.nodes[0], "owned")
    assert directory.exists(), "refused cleanup must preserve the unproven mailbox"


def test_bounded_mailbox_path_and_private_parent_cleanup(tmp_path):
    binding = scope()
    assert len(str(probe.mailbox_path(binding, binding.nodes[0]))) < 100
    child = tmp_path / "parent" / "child"
    child.mkdir(parents=True)
    probe.remove_empty_parent(child)
    assert child.exists(), "other owned mailboxes must not be removed"
    with pytest.raises(FileNotFoundError):
        probe.remove_empty_parent(tmp_path / "absent" / "child")


@pytest.mark.parametrize("action", ["cleanup-mailbox", "status", "daemon", "unknown"])
def test_node_probe_entry_uses_scoped_functions_only(monkeypatch, capsys, action):
    binding = scope()
    request = {"scope": binding.model_dump(mode="json"), "node": binding.nodes[0].name}
    calls = []
    if action == "daemon":
        request["daemon"] = True
    elif action != "unknown":
        request.update(action=action, payload={"pod_uid": "owned"})
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    monkeypatch.setattr(
        probe.NodeProbe, "run_daemon", lambda self: calls.append("daemon")
    )
    monkeypatch.setattr(probe, "cleanup_mailbox", lambda *a: calls.append("cleanup"))
    monkeypatch.setattr(
        probe,
        "daemon_request",
        lambda *a: {"scope_sha256": binding.digest(), "phase": "ATTACHED"},
    )
    result = probe.main()
    assert result == (1 if action == "unknown" else 0)
    output = capsys.readouterr().out
    if action == "unknown":
        assert json.loads(output) == {"error_kind": "BoundaryDenied"}
    elif action == "daemon":
        assert calls == ["daemon"] and output == ""
    elif action == "cleanup-mailbox":
        assert calls == ["cleanup"] and json.loads(output)["mailbox_absent"] is True
    else:
        assert json.loads(output)["phase"] == "ATTACHED"


@pytest.mark.parametrize("action", [None, "status"])
def test_expired_probe_entry_never_opens_a_host_channel(monkeypatch, capsys, action):
    binding = scope()
    binding = binding.model_copy(
        update={
            "maintenance_start": binding.maintenance_start - timedelta(hours=2),
            "maintenance_end": binding.maintenance_end - timedelta(hours=2),
        }
    )
    request = {"scope": binding.model_dump(mode="json"), "node": binding.nodes[0].name}
    if action:
        request.update(action=action, payload={})
    monkeypatch.setattr(probe.sys, "stdin", io.StringIO(json.dumps(request)))
    assert probe.main() == 1
    assert set(json.loads(capsys.readouterr().out)) == {"error_kind"}
